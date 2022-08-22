from torch import nn
from torch import optim
import torch
import matplotlib.pyplot as plt
from lib.utils import linear_norm, save_log, get_log, ensure_torch
from copy import deepcopy as dcp
from numpy import arange
from numpy.random import shuffle as shuffle_array

class AE(nn.Module):
    def __init__(self, num_features = 4, input_size = 200, num_units = 128, device = "cpu", architecture = None, norm = linear_norm, denorm = linear_norm, apply_basis_change_on_features = False, activation_fn = nn.ReLU(), feature_activation = linear_norm, out_fn = linear_norm,*args, **kwargs):
        super().__init__()
        self.norm = norm
        self.denorm = denorm
        self.num_features = num_features
        self.architecture = architecture
        self.num_units = num_units
        self.activation_fn = activation_fn
        self.device = device
        if architecture != None:
            if len(architecture["units"]) != 0:
                encoder_layers = [nn.Linear(input_size, architecture["units"][0]), activation_fn]
                for i in range(1,len(architecture["units"])):
                    encoder_layers.append(nn.BatchNorm1d(architecture["units"][i-1]))
                    encoder_layers.append(nn.Linear(architecture["units"][i-1], architecture["units"][i]))
                    encoder_layers.append(self.activation_fn)
                encoder_layers.append(nn.Linear(architecture["units"][-1], num_features))
                self.encoder = nn.Sequential(*encoder_layers).to(self.device)

                decoder_layers = [nn.Linear(num_features, architecture["units"][-1]), activation_fn]
                for i in range(1,len(architecture["units"])):
                    decoder_layers.append(nn.BatchNorm1d(architecture["units"][-i]))
                    decoder_layers.append(nn.Linear(architecture["units"][-i], architecture["units"][-i-1]))
                    decoder_layers.append(self.activation_fn)
                decoder_layers.append(nn.Linear(architecture["units"][0], input_size))
                self.decoder = nn.Sequential(*decoder_layers).to(self.device)
            else:
                encoder_layers = [nn.Linear(input_size, num_features)]
                self.encoder = nn.Sequential(*encoder_layers).to(self.device)
                decoder_layers = [nn.Linear(num_features, input_size)]
                self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(
                    in_features=input_size, out_features=num_units
                ),
                nn.BatchNorm1d(num_units),
                self.activation_fn,
                nn.Linear(
                    in_features=num_units, out_features=num_units//4
                ),
                nn.BatchNorm1d(num_units//4),
                self.activation_fn,
                nn.Linear(
                    in_features=num_units//4, out_features=num_features
                )
            ).to(self.device)

            self.decoder = nn.Sequential(
                nn.Linear(
                    in_features=num_features, out_features=num_units//4
                ),
                self.activation_fn,
                nn.BatchNorm1d(num_units//4),
                nn.Linear(
                    in_features=num_units//4, out_features=num_units
                ),
                self.activation_fn,
                nn.BatchNorm1d(num_units),
                nn.Linear(
                    in_features=num_units, out_features = input_size
                )
            ).to(self.device)

        self.feature_activation = feature_activation
        self.out_fn = out_fn
        self.device = device
        self.input_size = input_size
        self.has_trained = False
        try:
            self.change_of_basis = kwargs["change_of_basis"].to(self.device)
        except:
            self.change_of_basis = torch.eye(self.num_features).to(self.device)
            self.apply_basis_change_on_features = apply_basis_change_on_features
            if self.apply_basis_change_on_features:
                self.change_of_basis += torch.randn_like(self.change_of_basis).to(self.device)
                self.change_of_basis /= torch.det(self.change_of_basis)
    

    def reset(self):
        self.apply(weight_reset)

    def forward(self, features):
        features = self.norm(features)
        code = self.encoder(features)
        #code = self.feature_activation(code)
        reconstructed = self.decoder(code)
        reconstructed = self.out_fn(reconstructed)
        reconstructed = self.denorm(reconstructed)
        return reconstructed
    
    def feature_fn(self, features):
        features = ensure_torch(features)
        if len(features.shape) == 1:
            features = features.view((1,-1))
        features = ensure_torch(self.norm(features))
        try:
            code = self.encoder(features)
        except:

            code = []
            for batch in range(0,features.shape[0],500):
                code.append(self.encoder(features[batch:batch+500].to(self.device)))
            code = torch.cat(code)
                    
        code = self.feature_activation(code)
        code = code.detach()
        code = code @ self.change_of_basis
        feature_vals =  code.detach().to('cpu')
        return feature_vals

    def set_data(self,simulated_samples, split = [0.8,1.0],test_split = False, *args, **kwargs):
        if not test_split:
            split[1] = 1.0
        
        self.train_dataset = simulated_samples[:int(split[0]*simulated_samples.shape[0]),:].float()
        self.val_dataset = simulated_samples[int(split[0]*simulated_samples.shape[0]):int(split[1]*simulated_samples.shape[0]),:].float()
        self.test_dataset = None
        if test_split:
            self.test_dataset =  simulated_samples[int(split[1]*simulated_samples.shape[0]):,:].float()

    def train(self,
                    epochs = 1000,
                    batch_size = 50,
                    shuffle = True,
                    lr = 1e-4,
                    criterion = nn.MSELoss(),
                    weight_decay = 0.00,
                    amsgrad = False,
                    clip_max_norm = None,
                    scheduler = torch.optim.lr_scheduler.StepLR,
                    scheduler_kwargs = {"gamma": 1.0, "step_size": 15},
                    patience = 10,
                    multi_reset = 0,
                    threshold_gain = 1.0,
                    verbose = 1,
                    rescale_loss = 1.0,
                    print_rate = 10,
                    return_val = False,
                    *args, **kwargs
    ):
        
        train_dataset = torch.utils.data.TensorDataset(
            dcp(self.train_dataset),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = min(batch_size, self.train_dataset.shape[0]),
            shuffle = shuffle,
            drop_last = True
        )

        val_dataset = torch.utils.data.TensorDataset(
            dcp(self.val_dataset),
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = min(batch_size, self.val_dataset.shape[0]),
            drop_last = True
        )

        # Set up the training scheme
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay = weight_decay, amsgrad = amsgrad)
        schedule = scheduler(optimizer, **scheduler_kwargs)
        
        best_model = dcp(self.state_dict())
        best_val_loss = 2**16
        best_train_loss = 2**16
        early_stopping_counter = 0
        reset_counter = 0
        prev_loss = 0
        
        val_history = []
        for epoch in range(epochs):
            loss = 0
            val_loss = 0
            val_outputs = None
            
            for i, batch in enumerate(train_loader):
                batch_x = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_x)
                train_loss = criterion(self.norm(outputs), self.norm(batch_x))
                train_loss.backward()
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = clip_max_norm)
                optimizer.step()
                loss += train_loss.item()
                batch_features = None
                outputs = None
                torch.cuda.empty_cache()
            loss = loss / (i-1)
            schedule.step()
            val_loss = 0
            val_size = 0
            for i, batch in enumerate(val_loader):
                val_outputs = self(batch[0].to(self.device))
                val_size += batch[0].shape[0]
                val_loss += criterion(val_outputs, batch[0].to(self.device)).detach().item() * batch[0].shape[0]
            if val_size == 0:
                val_size = 1
            val_loss /= val_size
            val_history.append([loss,val_loss])
            val_outputs = None
            #val_outputs = None
            if verbose == 1 and (epoch) % print_rate == 0:
                print("epoch : {}, train loss = {:.6f}/{:.6f}, val loss = {:.6f}/{:.6f}, impatience: {}, resets left: {}         ".format(
                                                                                                                       epoch + 1,
                                                                                                                       loss * rescale_loss, best_train_loss * rescale_loss,
                                                                                                                       val_loss * rescale_loss, best_val_loss * rescale_loss,
                                                                                                                       early_stopping_counter, (multi_reset - reset_counter)
                                                                                                                       ),
                      end="\r", flush=True
                )
            if epoch == 0:
                best_model = dcp(self.state_dict())
                best_model_optim = dcp(optimizer.state_dict())
                best_val_loss = val_loss
                best_train_loss = loss
                continue
            # Update best model
            if best_val_loss > val_loss:
                best_model = dcp(self.state_dict())
                best_model_optim = dcp(optimizer.state_dict())
                diff_gain = 100 * (best_val_loss-val_loss) / best_val_loss
                if diff_gain > threshold_gain:
                    early_stopping_counter = 0
                    reset_counter = 0
                else:
                    early_stopping_counter += 1
                best_val_loss = val_loss
                best_train_loss = loss
            else: 
                early_stopping_counter += 1
                # Restart to best state including optimizer state (useful only with varying lr from scheduler)
                if early_stopping_counter > patience:
                    if reset_counter >= multi_reset:
                        break
                    else:
                        self.load_state_dict(dcp(best_model))
                        optimizer.load_state_dict(dcp(best_model_optim))
                        reset_counter+=1
                        early_stopping_counter = 0
        self.load_state_dict(best_model)
        self.has_trained = True
        self.zero_grad()
        if verbose != 0:
            print("\n")
        if return_val:
            return val_history
    
    def save(self, model_path, file_name, enforce_replace = False):
        state_dicts =  self.state_dict()
        for sdict in state_dicts:
            state_dicts[sdict] = state_dicts[sdict].to('cpu')
        log = {'state_dict': state_dicts,
               "init_args":{
                   "num_features":self.num_features,
                   "input_size":self.input_size,
                   "num_units":self.num_units, 
                   "architecture":self.architecture,
                   "apply_basis_change_on_features": self.apply_basis_change_on_features,
                   "norm":self.norm,
                   "denorm":self.denorm,
                   "activation_fn":self.activation_fn,
                   "change_of_basis": self.change_of_basis.to('cpu')
               }
              }
        save_log(log,model_path,file_name, enforce_replace=enforce_replace)

    def load(self, model_path, file_name, device = "cpu", *args, **kwargs):
        log = get_log(model_path,file_name)
        log['init_args']['device'] = device
        self.__init__(**log['init_args'])
        self = self.to(device)
        state_dicts =  log['state_dict']
        for sdict in state_dicts:
            log['state_dict'][sdict] = log['state_dict'][sdict].to(device)
        
        self.load_state_dict(log['state_dict'])    
        
    def get_type(self):
        return "AE"

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
