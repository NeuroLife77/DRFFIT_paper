from numpy import round as nround
from numpy import zeros as nzeros
from numpy import pi as npi
from numpy import where as nwhere
from numpy import exp as nexp
from numpy import abs as nabs
from numpy import empty as nempty
from numpy.random import seed as nseed
from numpy import array as narray
from numpy import linspace as nlinspace
from numpy.random import normal as nrandomnormal
import numba
from lib.HH_model.parameters_info import *
from torch import as_tensor
from lib.utils import ensure_numpy, ensure_torch
from scipy import stats as spstats
from torch import cat as tcat

@numba.njit
def syn_current(length, dt, t_start, curr_level):
    duration = length*1000
    t_on = t_start*1000
    t_off = duration - t_on
    
    # external current
    A_soma = npi * ((70.0 * 1e-4) ** 2)  # cm2
    I = nzeros((int(duration/dt),curr_level.shape[0]))
    I[int(nround(t_on / dt)) : int(nround(t_off / dt))] = (
        curr_level / A_soma
    )  # muA/cm2

    return I, [t_on, t_off]

@numba.njit
def efun(z):
    smaller  = 1 - z / 2
    bigger = z / (nexp(z) - 1)
    val = nwhere(nabs(z) < 1e-4, smaller, bigger)
    return val
@numba.njit
def alpha_m(x,param):
    v1 = x - param - 13.0
    return 0.32 * efun(-0.25 * v1) / 0.25
@numba.njit
def beta_m(x,param):
    v1 = x - param - 40
    return 0.28 * efun(0.2 * v1) / 0.2
@numba.njit
def alpha_h(x,param):
    v1 = x - param - 17.0
    return 0.128 * nexp(-v1 / 18.0)
@numba.njit
def beta_h(x,param):
    v1 = x - param - 40.0
    return 4.0 / (1 + nexp(-0.2 * v1))
@numba.njit
def alpha_n(x,param):
    v1 = x - param - 15.0
    return 0.032 * efun(-0.2 * v1) / 0.2
@numba.njit
def beta_n(x,param):
    v1 = x - param - 10.0
    return 0.5 * nexp(-v1 / 40)

# steady-states and time constants
@numba.njit
def tau_n(x,param):
    return 1 / (alpha_n(x,param) + beta_n(x,param))
@numba.njit
def n_inf(x,param):
    return alpha_n(x,param) / (alpha_n(x,param) + beta_n(x,param))
@numba.njit
def tau_m(x,param):
    return 1 / (alpha_m(x,param) + beta_m(x,param))
@numba.njit
def m_inf(x,param):
    return alpha_m(x,param) / (alpha_m(x,param) + beta_m(x,param))
@numba.njit
def tau_h(x,param):
    return 1 / (alpha_h(x,param) + beta_h(x,param))
@numba.njit
def h_inf(x,param):
    return alpha_h(x,param) / (alpha_h(x,param) + beta_h(x,param))

# slow non-inactivating K+
@numba.njit
def p_inf(x):
    v1 = x + 35.0
    return 1.0 / (1.0 + nexp(-0.1 * v1))
@numba.njit
def tau_p(x,param):
    v1 = x + 35.0
    return param / (3.3 * nexp(0.05 * v1) + nexp(-0.05 * v1))

@numba.njit
def simulate_HH(parameters, length: int = 0.2, dt: int = 0.0005, noise_seed = 42, initial_conditions = narray([-70]), curr_start = 0.005):
    num_sim = parameters.shape[0]
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    nseed(noise_seed)
    
    I, stat_info = syn_current(length, dt, curr_start, params[11])
    

    # Equivalent to allocating memory
    time_series_V = nempty((int(1000/dt*length),int(num_sim)))
    time_series_m_temp = nempty((1,int(num_sim)))
    time_series_n_temp = nempty((1,int(num_sim)))
    time_series_h_temp = nempty((1,int(num_sim)))
    time_series_p_temp = nempty((1,int(num_sim)))

    # Set initial conditions
    time_series_V[0] = initial_conditions[0]
    time_series_n_temp[0] = n_inf(time_series_V[0],params[1])
    time_series_m_temp[0] = m_inf(time_series_V[0],params[1])
    time_series_h_temp[0] = h_inf(time_series_V[0],params[1])
    time_series_p_temp[0] = p_inf(time_series_V[0])
    
    
    # Forward heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward euler
        tau_V_inv = (
            (time_series_m_temp[0] ** 3) * params[5] * time_series_h_temp[0]
            + (time_series_n_temp[0] ** 4) * params[3]
            + params[7]
            + params[2] * time_series_p_temp[0]
        ) / params[0]
        V_inf = (
            (time_series_m_temp[0] ** 3) * params[5] * time_series_h_temp[0] * params[6]
            + (time_series_n_temp[0] ** 4) * params[3] * params[4]
            + params[7] * params[8]
            + params[2] * time_series_p_temp[0] * params[4]
            + I[i]
            + params[10] * nrandomnormal(0,1,size=num_sim) / (dt ** 0.5)
        ) / (tau_V_inv * params[0])
        time_series_V[i+1] = V_inf + (time_series_V[i]-V_inf) * nexp(-dt * tau_V_inv)
        time_series_n_temp[0] = n_inf(time_series_V[i+1],params[1]) + (time_series_n_temp[0] - n_inf(time_series_V[i+1],params[1])) * nexp(-dt / tau_n(time_series_V[i+1],params[1]))
        time_series_m_temp[0] = m_inf(time_series_V[i+1],params[1]) + (time_series_m_temp[0] - m_inf(time_series_V[i+1],params[1])) * nexp(-dt / tau_m(time_series_V[i+1],params[1]))
        time_series_h_temp[0] = h_inf(time_series_V[i+1],params[1]) + (time_series_h_temp[0] - h_inf(time_series_V[i+1],params[1])) * nexp(-dt / tau_h(time_series_V[i+1],params[1]))
        time_series_p_temp[0] = p_inf(time_series_V[i+1]) + (time_series_p_temp[0] - p_inf(time_series_V[i+1])) * nexp(-dt / tau_p(time_series_V[i+1],params[9]))
    
    return time_series_V, I, stat_info

def HH_simulator(parameters, length: int = 0.1, dt: int = 0.0005, noise_seed = 42, initial_conditions = narray([-70]), curr_start = 0.005, get_summary_stats = False):
    length += curr_start
    parameters = ensure_numpy(parameters)
    V, I, current_info = simulate_HH(parameters, length = length, dt = dt, noise_seed = noise_seed, initial_conditions = initial_conditions, curr_start = curr_start)
    return ensure_torch(as_tensor(V.T)), ensure_torch(as_tensor(I.T))

