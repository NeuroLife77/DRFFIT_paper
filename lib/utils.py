import torch
import pickle as pkl
from numpy import printoptions
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch import as_tensor,cat,ones
from torch import max as tmax
import numpy as np

def set_mpl():
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['figure.facecolor'] = 'w'
#set_mpl()
def combine_datasets(dataset_list):
    # Takes a list of [path,file_name]
    combined_dataset = {}
    for i in range(len(dataset_list)):
        test_log_info =  get_log(f'{dataset_list[i][0]}',f'{dataset_list[i][1]}')
        for info in test_log_info:
            combined_dataset[info] = {}
    combined_dataset['data'] = {'x':[],'theta':[]}
    for i in range(len(dataset_list)):
        test_log_info =  get_log(f'{dataset_list[i][0]}',f'{dataset_list[i][1]}')
        for info in test_log_info:
            if info != 'data':
                combined_dataset[info][f'train_data_{i}'] = test_log_info[info]
            else:
                combined_dataset[info]['x'].append(test_log_info[info]['x'])
                combined_dataset[info]['theta'].append(test_log_info[info]['theta'])
    combined_dataset['data']['x'] = torch.cat(combined_dataset['data']['x'],dim = 0)
    combined_dataset['data']['theta'] = torch.cat(combined_dataset['data']['theta'],dim = 0)
    return combined_dataset

def load_dataset(path):
    
    try:
        with open(f'{path}log_file.pkl', 'rb') as f:
            log_info = pkl.load(f)
    except:
        log_info = None
    
    try:
        simulated_samples = torch.load(f'{path}simulated_samples.pt')
    except:
        simulated_samples = None

    try:
        parameters_samples = torch.load(f'{path}parameter_samples.pt')
    except:
        parameters_samples = None

    return simulated_samples, parameters_samples, log_info

def get_data(path, display_info = True):
    try:
        with open(f'{path}', 'rb') as f:
            log_info = pkl.load(f)
    except:
        print('Could not get data')
        return None
    if display_info:
        data_info(log_info)
    return log_info

def data_info(data):
    with printoptions(precision=8, suppress=True, edgeitems = 5 ,linewidth = 150, threshold=20):
        for info in data:
            if info == 'data' or info == 'target':
                print(f"\t{info}: x {data[info]['x'].shape}, theta {data[info]['theta'].shape}")
                continue
            if info == 'initial_conditions' or info == 'target':
                continue
            try:
                data_info(data[info])
            except:
                print(f"\t{info}: {data[info]}")

def search_log_info(data, info_log = "Log", show_shape = True, running_tabs = '', show_pars = False):
    for info in data:
        _ = data[info]
    print(f"{running_tabs}{info_log}:")
    running_tabs += '\t'
    with printoptions(precision=8, suppress=True, edgeitems = 5 ,linewidth = 150, threshold=20):
        for info in data:
            if info == 'data' or info == 'target':
                print(f"{running_tabs}{info}: x {data[info]['x'].shape}, theta {data[info]['theta'].shape}")
                continue
            if info == 'fit_history':
                print(f"{running_tabs}{info}: ")
                print(f"{running_tabs} x: {[data[info]['x'][i].shape for i in range(len(data[info]['x']))]}")
                print(f"{running_tabs} theta: {[data[info]['theta'][i].shape for i in range(len(data[info]['theta']))]}")
                print(f"{running_tabs} error: {[data[info]['error'][i] for i in range(len(data[info]['error']))]}")
                continue
            try:
                search_log_info(data[info], info_log = info,running_tabs=running_tabs)
            except:
                if show_shape:
                    try:
                        if data[info].shape[0]>1 or data[info].shape[1]>1 :
                            print(f"{running_tabs}{info}: {data[info].shape}")
                            continue
                    except:
                        try:
                            for item in data[info]:
                                try:
                                    search_log_info(data[item], info_log = info,running_tabs=running_tabs)
                                except:
                                    pass
                        except:
                            pass
                if not show_pars:
                    try:
                        print(f"{running_tabs}{info}: {torch.cat(data[info]).shape}")
                    except:
                        print(f"{running_tabs}{info}: {data[info]}")
                else:
                    print(f"{running_tabs}{info}: {data[info]}")
def mse(target, data, dim = 1):
    x = torch.as_tensor(data).float()
    x_target = torch.as_tensor(target).float()
    return torch.mean((x-x_target)**2, dim = dim)

def correlation_loss_fn(target,candidates):
    target = target.view(1,-1)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(target - target.mean(dim=1,keepdim=True),candidates - candidates.mean(dim=1,keepdim=True))
    return -pearson

def save_log(log, directory, file_name, enforce_replace = False):
    try:
        with open(f'{directory}/{file_name}.pkl', 'rb') as f:
            _ = pkl.load(f)
        if not enforce_replace:
            print("This file already exists, please select 'enforce_replace = True' to rewrite it.")
            return
    except:
        pass
    with open(f'{directory}{file_name}.pkl', 'wb') as f:
        pkl.dump(log, f)

def save_search_log(log, directory, file_name, enforce_replace = False):
    try:
        with open(f'{directory}/{file_name}.pkl', 'rb') as f:
            _ = pkl.load(f)
        if not enforce_replace:
            print("This file already exists, please select 'enforce_replace = True' to rewrite it.")
            raise
    except:
        pass
    with open(f'{directory}/{file_name}.pkl', 'wb') as f:
        pkl.dump(log, f)
    with open(f'{directory}/{file_name}_summary.pkl', 'wb') as f:
        pkl.dump(log["summary"], f)
        
def get_log(directory, file_name):
    if directory is None and 'drffit_object' in file_name:
        return None
    try:
        with open(f'{directory}{file_name}.pkl', 'rb') as f:
            log = pkl.load(f)
        return log
    except:
        print(f"File '{directory}{file_name}.pkl' does not exist.")
        return None

def get_search_log(directory, file_name, include_details = False):
    try:
        with open(f'{directory}{file_name}.pkl', 'rb') as f:
            log = pkl.load(f)
        if not include_details:
            for i in range(log['num_rounds']):
                try:
                    log[f'round_{i}']['data'] = torch.tensor([0])
                except:
                    pass
            log['fit_history']['x'] = torch.tensor([0])
        return log
    except:
        print(f"File '{directory}{file_name}.pkl' does not exist.")
        return None
    
def psd_norm(psd):
    return psd/0.0015
def psd_denorm(psd):
    return psd*0.0015

def max_norm(x):
    try:
        x /= x.amax(1, keepdim = True)
    except:
        x /= x.amax()
    return x
def linear_norm(x):
    return x

def scale_adjusted_rmse_for_PSD_AE(x,x_hat):
    rmse_dim1 = torch.sqrt(mse(x,x_hat))[:,0].double()
    adjusted_scale_rmse = rmse_dim1/x.amax(1)
    selected_adjusted_scale_rmse = torch.where(x.amax(1)>1, adjusted_scale_rmse, rmse_dim1)
    return selected_adjusted_scale_rmse.mean().float()

def custom_loss_adjusted_norm_rmse(target, data, dim = 1):
    x = torch.as_tensor(data).double() / 0.0015
    x_target = torch.as_tensor(target).double() / 0.0015
    mse = torch.mean((x-x_target)**2, dim = dim)
    rmse_dim1 = torch.sqrt(mse).double()
    adjusted_scale_rmse = rmse_dim1/x.amax(1)
    selected_adjusted_scale_rmse = torch.where(x.amax(1)>1, adjusted_scale_rmse, rmse_dim1)
    return selected_adjusted_scale_rmse.float().view(-1,1)


def get_real_individual_PSD_scaled(cutoff = 4, Hz = 50):
    real_data = []
    for i in range(2):
        real_data.append(np.load(f'../Data/PSD_dataset/reordered_dataset{i}.npy', allow_pickle=True))
    real_data_1 = as_tensor(real_data[0]).view(1,49,68,-1)
    real_data_2 = as_tensor(real_data[1]).view(1,49,68,-1)
    real_data = cat((real_data_1,real_data_2), dim = 0)
    for i in range(49):
        real_data[0][i] /= tmax(real_data[0][i][:,cutoff:2*Hz])
        real_data[1][i] /= tmax(real_data[1][i][:,cutoff:2*Hz])
    return real_data[:,:,:,cutoff:2*Hz]
full_real_psds = get_real_individual_PSD_scaled()

def ensure_torch(x, type_float = True):
    try:
        x = torch.as_tensor(x)
        if type_float:
            x = x.float()
    except:
        try:
            x = torch.from_numpy(x)
        except: 
            pass
    if type_float:
        try:
            x = x.float()
        except:
            pass
    return x

def ensure_numpy(x):
    
    try:
        x = x.detach()
    except:
        pass
    
    try:
        x = x.to('cpu')
    except:
        pass
    
    try:
        x = x.numpy()
    except:
        pass
    
    return x


def plot_fn(x, sqrt_fn = True, log_fn = False):
    if sqrt_fn:
        x = np.sqrt(x)
    if log_fn:
        x = np.log10(x)
    return x


def get_all_errors(dataset,num_rounds,num_samples,baseline_mean = None):
    append_to_baseline_mean = False
    if baseline_mean is None:
        baseline_mean = []
        append_to_baseline_mean = True
    means = []
    stds = []
    for i in range(len(dataset)):
        try:
            if append_to_baseline_mean:
                baseline_mean.append([0 for i in range(num_rounds)])
            _, temp_mean, temp_std = get_errors(dataset[i], num_rounds,num_samples, baseline_mean = baseline_mean[i])
            means.append(temp_mean.reshape(1,num_rounds))
            stds.append(temp_std.reshape(1,num_rounds))
        except:
            pass
    means = (np.concatenate(means, axis=0).mean(0))  
    stds = (np.concatenate(stds, axis=0).mean(0))  
    return means, stds

def get_errors(err, num_rounds, num_samples, baseline_mean = None):
    if baseline_mean is None:
        baseline_mean = [0 for i in range(num_rounds)]
    all_err = []
    err_mean = []
    err_std = []
    for i in range(num_rounds):
        try:
            round_err = np.array([err[f"fit_history"]['error'][j][i].data.item() for j in range(num_samples)]) - baseline_mean[i]
        except:
            round_err = np.array([err[f"fit_history"]['error'][j][i] for j in range(num_samples)]) - baseline_mean[i]
        round_err_large = round_err>1
        round_err[round_err_large] = 5e-8
        round_err = round_err.reshape((1,-1))
        err_mean.append(round_err.mean(1))
        err_std.append(round_err.std(1))
        all_err.append(round_err)
    err_std =  (np.concatenate(err_std, axis=0))
    err_mean =  (np.concatenate(err_mean, axis=0)) 
    return (all_err,err_mean, err_std)
def get_all_errors_old(dataset,num_rounds,num_samples,baseline_mean = None):
    append_to_baseline_mean = False
    if baseline_mean is None:
        baseline_mean = []
        append_to_baseline_mean = True
    means = []
    stds = []
    for i in range(len(dataset)):
        try:
            if append_to_baseline_mean:
                baseline_mean.append([0 for i in range(num_rounds)])
            _, temp_mean, temp_std = get_errors(dataset[i], num_rounds,num_samples, baseline_mean = baseline_mean[i])
            means.append(temp_mean.reshape(1,num_rounds))
            stds.append(temp_std.reshape(1,num_rounds))
        except:
            pass
    means = (np.concatenate(means, axis=0).mean(0))  
    stds = (np.concatenate(stds, axis=0).mean(0))  
    return means, stds

def get_errors_old(err, num_rounds, num_samples, baseline_mean = None):
    if baseline_mean is None:
        baseline_mean = [0 for i in range(num_rounds)]
    all_err = []
    err_mean = []
    err_std = []
    #print(baseline_mean)
    for i in range(num_rounds):
        try:
            round_err = np.array([err[j][f"fit_history"]['error'][i].data.item() for j in range(num_samples)]) - baseline_mean[i]
        except:
            round_err = np.array([err[j][f"fit_history"]['error'][i] for j in range(num_samples)]) - baseline_mean[i]
        round_err_large = round_err>1
        round_err[round_err_large] = 5e-8
        round_err = round_err.reshape((1,-1))
        #round_err = round_err[round_err<1].reshape((1,-1))
        err_mean.append(round_err.mean(1))
        err_std.append(round_err.std(1))
        all_err.append(round_err)
    #all_err =  np.concatenate(all_err,axis=0)  
    err_std =  (np.concatenate(err_std, axis=0))
    err_mean =  (np.concatenate(err_mean, axis=0)) 
    return (all_err,err_mean, err_std)

def plot_fn_reconstruction(testing_x,reconstructed_train_x, reconstructed_SE = None, freq = [0,100], rescale_plot = 0.0015, fn_name = 'fn', legend_args = {'ncol': 2,'prop':{'size':12}}, use_log = False, fig_size = (16,8)):
    set_mpl()
    testing_x = ensure_numpy(testing_x)
    reconstructed_train_x = ensure_numpy(reconstructed_train_x)
    if reconstructed_SE is not None:
        reconstructed_SE = ensure_numpy(reconstructed_SE)
        mean_reconstructed_SE = reconstructed_SE.mean(0)/rescale_plot
        std_reconstructed_SE = reconstructed_SE.std(0)/rescale_plot
    mean_testing_x = testing_x.mean(0)/rescale_plot
    std_testing_x = testing_x.std(0)/rescale_plot
    mean_reconstructed = reconstructed_train_x.mean(0)/rescale_plot
    std_reconstructed = reconstructed_train_x.std(0)/rescale_plot

    fig = plt.figure(figsize=fig_size)

    ax = plt.subplot(2,2,1)
    plt.plot(freq,reconstructed_train_x.max(0)/rescale_plot, label = f"Reconstructed {fn_name}")
    if reconstructed_SE is not None:
        plt.plot(freq,reconstructed_SE.max(0)/rescale_plot, label = f"Reconstructed SE {fn_name}")
    plt.plot(freq,testing_x.max(0)/rescale_plot, label = "Original")
    
    plt.legend(**legend_args)
    plt.title("Train data max power")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    ax = plt.subplot(2,2,2)
    plt.plot(freq,mean_reconstructed, label = f"Reconstructed {fn_name}")
    plt.fill_between(freq,mean_reconstructed+std_reconstructed,mean_reconstructed-std_reconstructed,alpha = 0.1)
    if reconstructed_SE is not None:
        plt.plot(freq,mean_reconstructed_SE, label = f"Reconstructed SE {fn_name}")
        plt.fill_between(freq,mean_reconstructed_SE+std_reconstructed_SE,mean_reconstructed_SE-std_reconstructed_SE,alpha = 0.25)
    plt.plot(freq,mean_testing_x, label = "Original")
    plt.fill_between(freq,mean_testing_x+std_testing_x,mean_testing_x-std_testing_x,alpha = 0.1)
    if not use_log:
        plt.ylim([0,None])
    plt.legend(**legend_args)
    plt.title("Train data mean power")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")


    error = (reconstructed_train_x/rescale_plot-testing_x/rescale_plot)**2
    max_error = np.sqrt(error.max(0))
    min_error = np.sqrt(error.min(0))
    mean_error = np.sqrt(error.mean(0))
    std_sample = np.sqrt(error.std(0))
    if reconstructed_SE is not None:
        error_SE = (reconstructed_SE/rescale_plot-testing_x/rescale_plot)**2
        max_error_SE = np.sqrt(error_SE.max(0))
        min_error_SE = np.sqrt(error_SE.min(0))
        mean_error_SE = np.sqrt(error_SE.mean(0))
        std_sample_SE = np.sqrt(error_SE.std(0))

    ax = plt.subplot(2,2,3)
    plt.plot(freq,max_error, label = "Max error")
    plt.plot(freq,min_error, label = "Min error")
    if reconstructed_SE is not None:
        plt.plot(freq,max_error_SE, label = "Max SE error")
        plt.plot(freq,min_error_SE, label = "Min SE error")
    plt.legend(**legend_args)
    plt.title("Bounds RMSE")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RMSE")

    ax = plt.subplot(2,2,4)
    plt.plot(freq,mean_error, label = "Mean error")
    plt.fill_between(freq,mean_error+std_sample,mean_error-std_sample,alpha = 0.15)
    if reconstructed_SE is not None:
        plt.plot(freq,mean_error_SE, label = "Mean error")
        plt.fill_between(freq,mean_error_SE+std_sample_SE,mean_error_SE-std_sample_SE,alpha = 0.15)
    plt.ylim([0,None])
    plt.legend(**legend_args)
    plt.title("Mean RMSE")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()
    
def plot_train_loss(val_hist,cutoff = 5, fig_size = (10,6), use_logscale = False):
    num_epochs = len(val_hist)
    ep = []
    for i in range(num_epochs):
        val_hist[i][0] = val_hist[i][0]
        val_hist[i][1] = val_hist[i][1]#.detach().to('cpu').data.item()
        ep.append(i)
    val_hist = ensure_numpy(np.array(val_hist)).T
    loss_label = 'Loss'
    if use_logscale:
        val_hist = np.log10(val_hist)
        loss_label += '(log_10)'
    fig = plt.figure(figsize=fig_size)
    ax = plt.subplot(1,1,1)
    plt.plot(ep[cutoff:],val_hist[0,cutoff:], label = "Train")
    plt.plot(ep[cutoff:],val_hist[1,cutoff:], label = "Val")
    plt.legend()
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel(loss_label)
    plt.show()