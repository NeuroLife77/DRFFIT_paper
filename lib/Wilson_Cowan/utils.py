import torch
from lib.utils import set_mpl

def clean_PSD(simulated_samples, parameters_samples_good, filter_cutoff = [1e-6, 0.00175], frequency_range = [None,None]):
    
    simulated_samples = ensure_torch(simulated_samples)
    parameters_samples_good = torch.as_tensor(parameters_samples_good)
    
    not_nan = torch.logical_not(simulated_samples.isnan().any(1))
    good_sim_low = simulated_samples[:, frequency_range[0]:frequency_range[1]].mean(1) > filter_cutoff[0]
    good_sim_high = simulated_samples[:, frequency_range[0]:frequency_range[1]].amax(1) < filter_cutoff[1]
    good_samples_scale = torch.logical_and(good_sim_low,good_sim_high)
    good_sim = torch.logical_and(not_nan,good_samples_scale)
    
    simulated_samples = simulated_samples[good_sim]
    parameters_samples_good = parameters_samples_good[good_sim]
    
    return simulated_samples, parameters_samples_good
    
def ensure_torch(x):
    try:
        x = torch.as_tensor(x)
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
