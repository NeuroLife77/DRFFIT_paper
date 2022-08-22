import numpy as np
from scipy import signal as sgl
import numba
from torch import as_tensor
from lib.utils import ensure_numpy, ensure_torch


@numba.njit
def simulate_heun_noise(parameters,
                        length: int = 302,
                        dt = 1.0,
                        num_sim: int = 1,
                        initial_conditions = np.array([0.25,0.25]),
                        noise_seed: int = 42,
                        store_I: bool = False,
                        precision: str = 'float32',
                        max_memory = 3.0,
                       ):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T.astype(precision)
    # Set seet
    np.random.seed(noise_seed)
    # White noise
    DE, DI = np.sqrt(2*params[-1] * dt).astype(precision) , np.sqrt(2*params[-2] * dt).astype(precision) 
    num_time_points = int(1000/dt*length)
    data_point_size = 4
    if precision != 'float32':
        data_point_size = 8
    expected_memory = data_point_size * (num_time_points) * num_sim 
    if store_I:
        expected_memory *= 2
    expected_memory += data_point_size * 6 * num_sim
    expected_memory /= 1e9
    if expected_memory > max_memory:
        raise
    # Equivalent to allocating memory
    time_series_E = np.empty((num_time_points,int(num_sim)), dtype = precision)
    time_series_I = np.zeros((4,int(num_sim)), dtype = precision)
    if store_I:
        time_series_I = np.empty((num_time_points,int(num_sim)), dtype = precision)
    time_series_E_temp = np.empty((1,int(num_sim)), dtype = precision)
    time_series_I_temp = np.empty((1,int(num_sim)), dtype = precision)
    time_series_E_corr = np.empty((1,int(num_sim)), dtype = precision)
    time_series_I_corr = np.empty((1,int(num_sim)), dtype = precision)
    time_series_E_noise = np.empty((1,int(num_sim)), dtype = precision)
    time_series_I_noise = np.empty((1,int(num_sim)), dtype = precision)
    # Set initial conditions
    time_series_E[0] = initial_conditions[0].astype(precision) 
    time_series_I[0] = initial_conditions[1].astype(precision) 
    # Heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(num_time_points-1):
        if store_I:
            j_0 = i
            j_1 = (i+1)
        else: # Using the I time series in a circular array if not stored
            j_0 = i%2
            j_1 = (i+1)%2
        # Forward Euler
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[j_0] + params[18] - params[9]
        time_series_I[j_1] = params[1] * time_series_E[i] - params[3] * time_series_I[j_0] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[j_1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[j_1] - params[11])))
        time_series_E[i+1] = dt * (((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[j_1] = dt * (((params[17] - params[15] * time_series_I[j_0]) * time_series_I[j_1]) - time_series_I[j_0]) / params[5] 
        time_series_E_noise = np.random.normal(0, 1, size = num_sim).astype(precision) *  DE 
        time_series_I_noise = np.random.normal(0, 1, size = num_sim).astype(precision) *  DI 
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] + time_series_E_noise
        time_series_I_temp = time_series_I[j_0] + time_series_I[j_1] + time_series_I_noise
        # Corrector point
        time_series_E_corr = params[0] * time_series_E_temp - params[2] * time_series_I_temp + params[18] - params[9]
        time_series_I_corr = params[1] * time_series_E_temp - params[3] * time_series_I_temp + params[19] - params[13]
        time_series_E_corr = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E_corr - params[7])))
        time_series_I_corr = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I_corr - params[11])))
        time_series_E_corr = dt*(((params[16] - params[14] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        time_series_I_corr = dt*(((params[17] - params[15] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr) / 2 + time_series_E_noise
        time_series_I[j_1] = time_series_I[j_0] + (time_series_I[j_1]+time_series_I_corr) / 2 + time_series_I_noise
        # Ceiling on activity of the neural mass
        time_series_E[i+1] = np.where(time_series_E[i+1] > 1.0, 1.0, time_series_E[i+1])
        time_series_I[j_1] = np.where(time_series_I[j_1] > 1.0, 1.0, time_series_I[j_1])
        # Floor on activity of the neural mass
        time_series_E[i+1] = np.where(time_series_E[i+1] < 0.0, 0.0, time_series_E[i+1])
        time_series_I[j_1] = np.where(time_series_I[j_1] < 0.0, 0.0, time_series_I[j_1])
        
    return time_series_E, time_series_I


def WC_stochastic_heun_PSD(
                               parameters,
                               length: int = 302,
                               dt: int = 1,
                               initial_conditions = None,
                               noise_seed: int = 42,
                               PSD_cutoff = [4,100],
                               out_tensor = True,
                               get_psd_I = False,
                               filter_bad = True,
                               remove_bad = False,
                               return_bool_bad = False
                          ):
    
    parameters = ensure_numpy(parameters)
    if initial_conditions is None:
        initial_conditions = np.random.rand(2, parameters.shape[0])
    time_series_E, time_series_I = simulate_heun_noise(
                                        parameters,
                                        length = length,
                                        dt = dt,
                                        num_sim = parameters.shape[0],
                                        noise_seed = noise_seed,
                                        initial_conditions = initial_conditions,
                                        store_I=get_psd_I,
                                    )
    
    bad_sims = np.any(
        (
                                np.any(np.isinf(time_series_E), axis = 0),
                                np.any(np.isnan(time_series_E), axis = 0),
                                np.any(np.isinf(time_series_I), axis = 0),
                                np.any(np.isnan(time_series_I), axis = 0)
        ), axis = 0
    )
    
    good_parameters = parameters

    if filter_bad:
        time_series_E[:,bad_sims] = np.random.rand(time_series_E[:,bad_sims].shape[0],time_series_E[:,bad_sims].shape[1]) * 1e-10
        time_series_I[:,bad_sims] = np.random.rand(time_series_I[:,bad_sims].shape[0],time_series_I[:,bad_sims].shape[1]) * 1e-10
    elif remove_bad:
        time_series_E = time_series_E[:,np.logical_not(bad_sims)]
        good_parameters = parameters[np.logical_not(bad_sims),:]
        time_series_I = time_series_I[:,np.logical_not(bad_sims)]
        try:
            faulty_dummy = ensure_torch(as_tensor([[-1 for i in range(PSD_cutoff[0], PSD_cutoff[1])]]))
            if time_series_E.shape[1] < 1:
                if get_psd_I:
                    return faulty_dummy, faulty_dummy, faulty_dummy, faulty_dummy
                else:
                    return faulty_dummy, faulty_dummy, faulty_dummy
        except:
            if get_psd_I:
                return faulty_dummy, faulty_dummy, faulty_dummy, faulty_dummy
            else:
                return faulty_dummy, faulty_dummy, faulty_dummy
    
    freq , psd_E = sgl.welch(time_series_E,fs=(1000/dt), nperseg=2000/dt, axis = 0)
    psd_E, freq = psd_E.T[:,PSD_cutoff[0]:PSD_cutoff[1]], freq[PSD_cutoff[0]:PSD_cutoff[1]]
    
    if get_psd_I:
        _ , psd_I = sgl.welch(time_series_I,fs=(1000/dt), nperseg=2000/dt, axis = 0)
        psd_I = psd_I.T[:,PSD_cutoff[0]:PSD_cutoff[1]]
        if out_tensor:
            psd_E, psd_I, freq, good_parameters = ensure_torch(psd_E), ensure_torch(psd_I), ensure_torch(freq), ensure_torch(good_parameters)
        return psd_E, psd_I, freq, good_parameters
    else:
        time_series_I = None
        if out_tensor:
            psd_E, freq, good_parameters = ensure_torch(psd_E), ensure_torch(freq), ensure_torch(good_parameters)
        if return_bool_bad:
            return psd_E, freq, good_parameters, ensure_torch(bad_sims)
        return psd_E, freq, good_parameters
    
    
    
    
########################################################################################### Other simulators ###########################################################################################

def compute_PSD(timeseries, dt):
    freq , psds = sgl.welch(timeseries,fs=(1000/dt), nperseg=2000/dt, axis = 0)
    return psds[:100,:].T, freq[:100]

@numba.njit 
def simulate_euler(parameters,length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25])):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Forward euler performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt * (((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt * (((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E[i+1] += time_series_E[i] 
        time_series_I[i+1] += time_series_I[i] 
    return time_series_E, time_series_I

@numba.njit
def simulate_heun(parameters, length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25])):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_E_temp = np.empty((1,int(num_sim)))
    time_series_I_temp = np.empty((1,int(num_sim)))
    time_series_E_corr = np.empty((1,int(num_sim)))
    time_series_I_corr = np.empty((1,int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Forward heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward euler
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt*(((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt*(((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] 
        time_series_I_temp = time_series_I[i] + time_series_I[i+1]
        # Corrector point
        time_series_E_corr = params[0] * time_series_E_temp - params[2] * time_series_I_temp + params[18] - params[9]
        time_series_I_corr = params[1] * time_series_E_temp - params[3] * time_series_I_temp + params[19] - params[13]
        time_series_E_corr = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E_corr - params[7])))
        time_series_I_corr = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I_corr - params[11])))
        time_series_E_corr = dt*(((params[16] - params[14] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        time_series_I_corr = dt*(((params[17] - params[15] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr)/2 
        time_series_I[i+1] = time_series_I[i] + (time_series_I[i+1]+time_series_I_corr)/2         
    return time_series_E, time_series_I

@numba.njit 
def simulate_euler_noise(parameters, length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25]), noise_seed: int = 42):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Set seed
    np.random.seed(noise_seed)
    # White noise
    DE, DI = np.sqrt(2*params[-1]), np.sqrt(2*params[-2])
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Forward euler performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt * (((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt * (((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E[i+1] += time_series_E[i] + np.random.normal(0,1,size=num_sim) *  DE
        time_series_I[i+1] += time_series_I[i] + np.random.normal(0,1,size=num_sim) *  DI 
    return time_series_E, time_series_I
