from torch import nn
from copy import deepcopy as dcp
import torch
import numpy as np
import warnings
from numpy import arange
from numpy.random import shuffle as shuffle_array
from lib.utils import *


class ResNet(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module = module
    def forward(self, inputs):
        return self.module(inputs) + inputs
               
            
    
class unit_net(nn.Module):
    def __init__(self, input_size = 24, num_features = 1,
                 architecture = {'units':[50 for i in range(2)],'skip_connection':False},
                 activation_fn = nn.ReLU(),
                 device = 'cpu',
                 *args, **kwargs):
        
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.num_features = num_features
        self.activation_fn = activation_fn
        self.skip_connect = False
        if "skip_connection" in architecture:
            if isinstance(architecture['skip_connection'], bool):
                self.skip_connect = architecture['skip_connection']
        if len(architecture["units"]) != 0:
            self.architecture = architecture
            self.all_blocks = [nn.Sequential(
                nn.Linear(input_size, architecture["units"][0]),
                nn.BatchNorm1d(architecture["units"][0]),
                self.activation_fn
            )]
            for i in range(1,len(architecture["units"])):
                if self.skip_connect:
                    if architecture["units"][i-1] == architecture["units"][i]:
                        self.all_blocks.append(ResNet(nn.Sequential(
                            nn.Linear(architecture["units"][i-1], architecture["units"][i]),
                            nn.BatchNorm1d(architecture["units"][i]),
                            self.activation_fn
                        )))
                    else:
                        self.all_blocks.append(nn.Sequential(
                            nn.Linear(architecture["units"][i-1], architecture["units"][i]),
                            nn.BatchNorm1d(architecture["units"][i]),
                            self.activation_fn
                        ))
                else:
                    self.all_blocks.append(nn.Sequential(
                        nn.Linear(architecture["units"][i-1], architecture["units"][i]),
                        nn.BatchNorm1d(architecture["units"][i]),
                        self.activation_fn
                    ))
            self.all_blocks.append(nn.Sequential(
                    nn.Linear(architecture["units"][-1], num_features),
                ))

            self.regression_net = nn.Sequential(*self.all_blocks).to(self.device)
                
    
    def forward(self, thetas):
        activation = self.regression_net(thetas)
        return activation
    
    def reset_parameters(self):
        for layer in self.regression_net:
            try:
                layer.reset_parameters()
            except:
                pass
    
def linear_fn(x):
    return x


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


class subspace_estimator:
    def __init__(self, theta_min, theta_range, input_size = 24, device = "cpu", store_data = True):
        self.num_pars = input_size
        self.device = device
        self.net_list = []
        self.net_list_units = []
        self.net_has_trained = []
        self.num_features = len(self.net_list)
        self.theta_min = ensure_torch(theta_min)
        self.theta_range = ensure_torch(theta_range)
        self.has_data = False
        self.store_data = store_data
        self.prev_data = {}
        self.curr_data = ""
        
        self.feature_mean = None
        self.feature_std = None
        self.train_thetas = None
        self.train_feature_set = None
        self.val_thetas = None
        self.val_feature_set = None
        self.test_thetas = None
        self.test_feature_set = None
        
    
    def add_features(self, num_features, num_units = [128], device = "cpu", combine = False, enforce_replace_all = False, *args, **kwargs):
        if enforce_replace_all:
            self.net_list = []
            self.net_list_units = []
            self.net_has_trained = []
        if len(num_units) == 1:
            units = num_units[0]
            num_units = [units for _ in range(num_features)] 
        if combine:
            self.net_list.append(unit_net(input_size=self.num_pars, num_features = num_features,
            num_units=num_units[0], device = device, **kwargs).to(device))
            self.net_has_trained.append(False)
        else:
            for i in range(num_features):
                self.net_list.append(unit_net(input_size=self.num_pars,num_units=num_units[i], device = device, **kwargs).to(device))
                self.net_has_trained.append(False)
        self.num_features = len(self.net_list)

    def add_feature(self, num_units = 128, device = "cpu", *args, **kwargs):
        self.net_list.append(unit_net(input_size=self.num_pars,num_units=num_units, device = device, **kwargs).to(device))
        self.net_has_trained.append(False)
        self.num_features = len(self.net_list)

    def reset_feature(self, i, num_units = None, *args, **kwargs):
        if num_units == None:
            self.net_list[i].reset_parameters()
            return
        prev_device = self.net_list[i].device
        self.net_list[i] = unit_net(input_size=self.num_pars, num_units=num_units, device = prev_device, **kwargs).to(prev_device)
            

    def add_custom_net_feature(self, net, has_trained = False,device = 'cpu' ,*args, **kwargs):
        self.net_list.append(net.to(device))
        self.net_has_trained.append(has_trained)
        self.num_features = len(self.net_list)

    def set_data(self, parameters_samples, feature_set, split = [0.8,1.0], test_split = False, data_name = "default", store = None, norm_features= False, *args, **kwargs):
        train_cutoff = int(split[0]*parameters_samples.shape[0])
        if test_split:
            val_cutoff = int(split[1]*parameters_samples.shape[0])
            self.test_split = True
        else:
            val_cutoff = -1
            self.test_split = False
        
        
        self.feature_mean = feature_set[:train_cutoff,:].mean(0)
        self.feature_std = feature_set[:train_cutoff,:].std(0)
        if norm_features:
            feature_set = (feature_set-self.feature_mean)/self.feature_std
        
        self.train_thetas = ensure_torch((parameters_samples[:train_cutoff,:]-self.theta_min)/self.theta_range)
        self.train_feature_set = ensure_torch(feature_set[:train_cutoff,:])
        self.val_thetas = ensure_torch((parameters_samples[train_cutoff:val_cutoff,:]-self.theta_min)/self.theta_range)
        self.val_feature_set = ensure_torch(feature_set[train_cutoff:val_cutoff,:])
        
        if test_split:
            self.test_thetas = ensure_torch((parameters_samples[val_cutoff:,:]-self.theta_min)/self.theta_range)
            self.test_feature_set = ensure_torch(feature_set[val_cutoff:,:])
        else:
            val_cutoff = -1
        self.has_data = True
        if store == None and self.store_data or store == True:
            self.prev_data[data_name] = {"parameters_samples":parameters_samples,"feature_set":feature_set,"split": split, "test_split":test_split}
            self.curr_data = data_name

    def set_prev_data(self, name, split = None, test_split = None, *args, **kwargs):
        try:
            data_set = self.prev_data[name]
        except:
            return None
        if split != None:
            data_set['split'] = split
        if test_split != None:
            data_set['test_split'] = test_split
        curr_data = self.curr_data
        self.set_data(**data_set)
        return curr_data
        
    def train(self,
                    index,
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
        # Get the model to train
        model_temp = self.net_list[index]
        self.net_has_trained[index] = True 
        # Define the features to train on
        if model_temp.num_features > 1:
            index = torch.arange(model_temp.num_features)
        
        # Set up the data loaders
        train_dataset = torch.utils.data.TensorDataset(
            dcp(self.train_thetas.to(model_temp.device)),
            dcp(self.train_feature_set[:,index].view(-1,model_temp.num_features).to(model_temp.device))
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = min(batch_size, self.train_thetas.shape[0]),
            shuffle = shuffle,
            drop_last = True
        )

        val_dataset = torch.utils.data.TensorDataset(
            dcp(self.val_thetas.to(model_temp.device)),
            dcp(self.val_feature_set[:,index].view(-1,model_temp.num_features).to(model_temp.device))
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = min(batch_size, self.val_thetas.shape[0]),
            drop_last = True
        )

        # Set up the training scheme
        optimizer = torch.optim.AdamW(model_temp.parameters(), lr=lr, weight_decay = weight_decay, amsgrad = amsgrad)
        schedule = scheduler(optimizer, **scheduler_kwargs)
        
        # Training variables to keep track of
        best_model = dcp(model_temp.state_dict())
        best_model_optim = dcp(optimizer.state_dict())
        best_val_loss = 2**16
        best_train_loss = 2**16
        early_stopping_counter = 0
        reset_counter = 0
        val_history = []
        # Training 
        for epoch in range(epochs):
            loss = 0
            for i, batch in enumerate(train_loader):
                batch_theta = batch[0]
                batch_target = batch[1]
                optimizer.zero_grad()
                outputs = model_temp(batch_theta)
                train_loss = criterion(outputs, batch_target)
                train_loss.backward()
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model_temp.parameters(), max_norm = clip_max_norm)
                optimizer.step()
                loss += train_loss.item()
                outputs = None
            schedule.step()
            loss = loss / (i-1)
            val_loss = 0
            val_size = 0
            for i, batch in enumerate(val_loader):
                val_outputs = model_temp(batch[0])
                val_size += batch[0].shape[0]
                val_loss += criterion(val_outputs, batch[1]).detach().item() * batch[0].shape[0]
            if val_size == 0:
                val_size = 1
            val_loss /= val_size
            val_history.append([loss, val_loss])
            val_outputs = None
            if verbose == 1 and (epoch) % print_rate == 0:
                print("epoch : {}, train loss = {:.6f}/{:.6f}, val loss = {:.6f}/{:.6f}, impatience: {}, resets left: {}         ".format(
                                                                                                                       epoch + 1,
                                                                                                                       loss * rescale_loss, best_train_loss * rescale_loss,
                                                                                                                       val_loss * rescale_loss, best_val_loss * rescale_loss,
                                                                                                                       early_stopping_counter, (multi_reset - reset_counter)
                                                                                                                       ),
                      end="\r", flush=True
                )
            # Initialize best model
            if epoch == 0:
                best_model = dcp(model_temp.state_dict())
                best_model_optim = dcp(optimizer.state_dict())
                best_val_loss = val_loss
                best_train_loss = loss
                continue
            # Update best model
            if best_val_loss > val_loss:
                best_model = dcp(model_temp.state_dict())
                best_model_optim = dcp(optimizer.state_dict())
                diff_gain = 100 * (best_val_loss-val_loss) / best_val_loss
                if diff_gain > threshold_gain:
                    early_stopping_counter = 0
                    reset_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter > patience:
                        if reset_counter >= multi_reset:
                            break
                        else:
                            reset_counter+=1
                            early_stopping_counter = 0
                best_val_loss = val_loss
                best_train_loss = loss
            else: 
                early_stopping_counter += 1
                # Restart to best state including optimizer state (useful only with varying lr from scheduler)
                if early_stopping_counter > patience:
                    if reset_counter >= multi_reset:
                        break
                    else:
                        old = dcp(best_model_optim)
                        old['param_groups'] =  optimizer.state_dict()['param_groups']
                        model_temp.load_state_dict(dcp(best_model))
                        optimizer.load_state_dict(old)
                        reset_counter+=1
                        early_stopping_counter = 0
        model_temp.load_state_dict(best_model)
        if verbose != 0:
            print("\n")
        if return_val:
            return val_history
        
    
    def train_all(self, features = None, skip = [], return_val = False,*args, **kwargs):
        """Wrapper function for the train function that allows training of all nets"""
        if features == None:
            features = [i for i in range(self.num_features)]
        val_hist = []
        for i in range(self.num_features):
            if i in skip:
                continue
            print(f'Feature {features[i]}:')
            if return_val:
                val_history = self.train(i, return_val = return_val,*args, **kwargs)
                val_hist.append(val_history)
            else:
                self.train(i, return_val = return_val,*args, **kwargs)
        if return_val:
            return val_hist
    
    def get_subspace(self, samples,  num_eigen_vect_per_feature = 1, *args, **kwargs):
       # if target is not None:
       #     return self.get_gradient_target_subspace(samples, target, num_eigen_vect_per_feature = 1, *args, **kwargs)
        drffit_eigen_basis = []
        for i in range(len(self.net_list)):
            drffit_eigen_basis.append(self.get_basis_vector(samples,i,num_eigen_vect=num_eigen_vect_per_feature))
        drffit_subspace = torch.cat(drffit_eigen_basis,dim = 0) 
        return drffit_subspace.detach().to("cpu")

    def get_basis_vector(self, samples, i, criterion_basis = nn.MSELoss(), num_eigen_vect = 1, *args, **kwargs):
        
        device = self.net_list[i].device
        norm_samples = dcp(((samples-self.theta_min)/self.theta_range)).to(device)
        drffit_eigen_basis = []
        norm_samples.requires_grad = True
        
        # Reset grad
        self.net_list[i].zero_grad()
        
        # Compute grad
        predictions = self.net_list[i](norm_samples)
        drffit_loss = criterion_basis(predictions, predictions.mean(0).repeat(predictions.shape[0],1))
        drffit_loss.backward()
        
        # Get gradient of input samples
        gradients = torch.squeeze(norm_samples.grad) 
        
        # Get the eigenvectors of the gradient over the samples
        outer_products = torch.einsum("bi,bj->bij", (gradients, gradients))
        average_outer_product = outer_products.mean(dim=0)
        eigen_values, eigen_vectors = torch.linalg.eigh(average_outer_product, UPLO="U")
        av_gradient = torch.mean(gradients, dim=0)
        av_gradient = av_gradient / torch.norm(av_gradient)
        av_eigenvec = torch.mean(eigen_vectors * eigen_values, dim=1)
        av_eigenvec = av_eigenvec / torch.norm(av_eigenvec)

        # Invert if the negative eigenvectors are closer to the average gradient.
        if (torch.mean((av_eigenvec - av_gradient) ** 2)) > (torch.mean((-av_eigenvec - av_gradient) ** 2)):
            eigen_vectors = -eigen_vectors
        drffit_eigen_basis.append([eigen_vectors,eigen_values])
        
        # Create the subspace from the basis vectors of the network i
        drffit_subspace = []
        for j in range(1,num_eigen_vect+1):
            drffit_subspace.append(drffit_eigen_basis[0][0][-j].view(1,-1))
        drffit_subspace = torch.cat(drffit_subspace,dim = 0) 
        
        # Ensure no memory leak
        norm_samples = None
        # Reset gradient for next call
        self.net_list[i].zero_grad(set_to_none = True)
        return drffit_subspace.detach().to('cpu')

    def probe_points_for_target(self, samples, target, which = None, *args, **kwargs):
        if which is None:
            which = [0]
        running_loss = []
        for i in which:
            device = self.net_list[i].device
            norm_samples = dcp(((samples-self.theta_min)/self.theta_range)).to(device)
            target_features = dcp(ensure_torch(target)).to(device).repeat(norm_samples.shape[0],1)
            self.net_list[i].zero_grad()
            predictions = self.net_list[i](norm_samples)
            loss = torch.mean((predictions - target_features)**2, dim = 1, keepdim = True)
            running_loss.append(loss)
            self.net_list[i].zero_grad()
        drffit_loss = torch.cat(running_loss, dim = 1).mean(1)
        candidates = torch.argsort(drffit_loss).detach().to('cpu')
        return samples[candidates]
    
    def get_gradient_target_subspace(self, samples, target, num_eigen_vect_per_feature = 1, *args, **kwargs):
        drffit_eigen_basis = []
        for i in range(len(self.net_list)):
            drffit_eigen_basis.append(self.get_feature_gradient_vector(samples, i, target, num_eigen_vect=num_eigen_vect_per_feature))
        drffit_gradient_subspace = torch.cat(drffit_eigen_basis,dim = 0) 
        return drffit_gradient_subspace.detach().to("cpu")
    
    def get_feature_gradient_vector(self, samples, i, target, num_eigen_vect = 1, *args, **kwargs):
        device = self.net_list[i].device
        target_features = dcp(target).to(device)
        norm_samples = dcp(((samples-self.theta_min)/self.theta_range)).to(device)
        drffit_eigen_basis = []
        criterion_basis = nn.MSELoss()
        norm_samples.requires_grad = True
        self.net_list[i].zero_grad()
        predictions = self.net_list[i](norm_samples)
        drffit_loss = criterion_basis(predictions, target_features.repeat(predictions.shape[0],1))
        drffit_loss.backward()
        gradients = torch.squeeze(norm_samples.grad) 
        outer_products = torch.einsum("bi,bj->bij", (gradients, gradients))
        average_outer_product = outer_products.mean(dim=0)
        eigen_values, eigen_vectors = torch.linalg.eigh(average_outer_product, UPLO="U")
        av_gradient = torch.mean(gradients, dim=0)
        av_gradient = av_gradient / torch.norm(av_gradient)
        av_eigenvec = torch.mean(eigen_vectors * eigen_values, dim=1)
        av_eigenvec = av_eigenvec / torch.norm(av_eigenvec)

        # Invert if the negative eigenvectors are closer to the average gradient.
        if (torch.mean((av_eigenvec - av_gradient) ** 2)) > (
            torch.mean((-av_eigenvec - av_gradient) ** 2)
        ):
            eigen_vectors = -eigen_vectors
        drffit_eigen_basis.append([eigen_vectors,eigen_values])
    
        drffit_subspace = []
        for j in range(1,num_eigen_vect+1):
            drffit_subspace.append(drffit_eigen_basis[0][0][-j].view(1,-1))
        drffit_gradient_subspace = torch.cat(drffit_subspace,dim = 0) 
        norm_samples = None
        self.net_list[i].zero_grad(set_to_none = True)
        return drffit_gradient_subspace.detach().to('cpu')
    
    def load_as_log(self, model_path, file_name, device = "cpu",load_data_to_device = False, *args, **kwargs):
        log = get_log(model_path, file_name)
        
        self.__init__(**log['init_args'])
        
        self.device = device
        self.net_list = []
        self.net_has_trained = log['state_info']['net_has_trained']
        self.num_features = log['state_info']['num_features']
        self.has_data = log['data']['has_data']
        self.prev_data = log['data']['prev_data']
        self.curr_data = log['data']['curr_data']
        if load_data_to_device:
            for info in log['data']:
                try:
                    log['data'][info] = log['data'][info].to(self.device)
                except:
                    pass
        self.feature_mean = log['data']['feature_mean']
        self.feature_std = log['data']['feature_std']
        self.train_thetas = log['data']['train_thetas']
        self.train_feature_set = log['data']['train_feature_set']
        self.val_thetas = log['data']['val_thetas']
        self.val_feature_set = log['data']['val_feature_set']
        self.test_thetas = log['data']['test_thetas']
        self.test_feature_set = log['data']['test_feature_set']
        try:
            # To ensure backwards compatibility (temporary)
            self.batch_size = log['data']['batch_size']
            self.batch_split = log['data']['batch_split']
            self.shuffle = log['data']['shuffle']
        except:
            pass
        
        nets =  log['nets']
        for net_log in nets:
            net_state_dict = net_log['state_dict']
            for sdict in net_state_dict:
                net_state_dict[sdict] = net_state_dict[sdict].to(device)
            net = unit_net(**net_log['init_args']).to(device)
            net.device = device
            net.load_state_dict(net_state_dict)
            self.net_list.append(net)
    
    def save_as_log(self, model_path, file_name, enforce_replace = False):
        log = {}
        nets = []
        for model in self.net_list:
            state_dicts = model.state_dict()
            for sdict in state_dicts:
                state_dicts[sdict] = state_dicts[sdict].to('cpu')
            net_log = {
                'state_dict': state_dicts,
                "init_args":{
                    "input_size":model.input_size,
                    "num_features":model.num_features,
                    "architecture": model.architecture,
                    "activation_fn": model.activation_fn,
               }
              }
            nets.append(net_log)
        log['nets'] = nets
        log['init_args'] = {
            'theta_min': self.theta_min,
            'theta_range': self.theta_range,
            'input_size': self.num_pars,
            "store_data": self.store_data,
        }
        log['data'] = {
            'prev_data': self.prev_data,
            'curr_data': self.curr_data,
            'has_data': self.has_data,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'train_thetas': self.train_thetas,
            'train_feature_set': self.train_feature_set,
            'val_thetas': self.val_thetas,
            'val_feature_set': self.val_feature_set,
            'test_thetas': self.test_thetas,
            'test_feature_set': self.test_feature_set,
        }
        for info in log['data']:
            try:
                log['data'][info] = log['data'][info].to("cpu")
            except:
                pass
        log['state_info'] = {
            'net_has_trained': self.net_has_trained,
            "num_features": self.num_features,
        }
        save_log(log, model_path, file_name, enforce_replace=enforce_replace)
    
    def get_all_state_dicts(self):
        return [model.state_dict() for model in self.net_list]
    def get_type(self):
        return "subspace_estimator"
