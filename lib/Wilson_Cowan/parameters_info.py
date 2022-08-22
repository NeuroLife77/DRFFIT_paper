import numpy as np

from torch import tensor
common_worst40 = tensor([4130, 3449,  559, 3443,  560,  117,  798, 1115, 3409,  593, 6305,   77,
         865, 6207, 3447,  866, 2900,  935,   83, 2911, 1098, 3413, 6203,  111,
        2973, 2866, 2871, 2907, 1820,  155,  941, 6241, 2909, 4241, 4236, 2875,
        4237, 3206, 2492, 5118])

avg_worst40 = tensor([ 117, 4130,  560,  798,  865,  559, 3449, 3443,  593, 1115,   83, 3409,
          77,  866, 6203, 6207,  111, 2871,  935, 3447, 2911, 6305, 1098, 2907,
        4241, 2900, 2909, 6241, 3658, 2866, 3413, 3622,  383, 2875, 2973,  326,
        6238,  941,   82,  155])

pars = {
    'c_ee': 16.0, #Ex-to-Ex coupling coefficient, index = 0
    'c_ei': 12.0, #Ex-to-In coupling coefficient, index = 1
    'c_ie': 15.0, #In-to-Ex coupling coefficient, index = 2
    'c_ii': 3.0, #In-to-In coupling coefficient, index = 3
    'tau_e': 8.0, #Ex membrane time-constant, index = 4
    'tau_i': 18.0, #In membrane time-constant, index = 5
    'a_e': 1.3, #Ex Value of max slope of sigmoid function (1/a_e) is related to variance of distribution of thresholds, index = 6
    'b_e': 4.0, #Sigmoid function threshold, index = 7
    'c_e': 1.0, #Amplitude of Ex response function, index = 8
    'theta_e': 0.0, #Position of max slope of S_e, index = 9
    'a_i': 2.0, #In Value of max slope of sigmoid function (1/a_e) is related to variance of distribution of thresholds, index = 10
    'b_i': 3.7, #Sigmoid function threshold, index = 11
    'c_i': 1.0, #Amplitude of In response function, index = 12
    'theta_i': 0.0, #Position of max slope of S_i, index = 13
    'r_e': 1.0, #Ex refractory period, index = 14
    'r_i': 1.0, #In refractory period, index = 15
    'k_e': 1.0, #Max value of the Ex response function, index = 16
    'k_i': 1.0, #Max value of the In response function, index = 17
    'P': 1.25, #Balance between Ex and In masses, index = 18
    'Q': 0.0, #Balance between Ex and In masses, index = 19
    'alpha_e': 1.0, #Balance between Ex and In masses, index = 20
    'alpha_i': 1.0, #Balance between Ex and In masses, index = 21

}
parameter_names = ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i', 'a_e', 'b_e', 'c_e', 'theta_e', 'a_i', 'b_i', 'c_i', 'theta_i', 'r_e', 'r_i', 'k_e', 'k_i', 'P', 'Q', 'alpha_e', 'alpha_i', 'noise_E','noise_I']

parameters_alpha_peak = np.array([1.6000e+01, 1.2000e+01, 1.5000e+01, 3.0000e+00, 8.0000e+00, 25.0000e+00,
        1.3000e+00, 4.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00, 3.7000e+00,
        1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.500e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,1.0000e-05,1.0000e-05])



parameters_original = np.array([1.6000e+01, 1.2000e+01, 1.5000e+01, 3.0000e+00, 8.0000e+00, 8.0000e+00,
        1.3000e+00, 4.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00, 3.7000e+00,
        1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.500e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e-05, 1.0000e-05])

parameters_lower_bound_narrow = np.array([
    1.1000e+01, #c_ee
    0.2000e+01, #c_ei
    0.2000e+01, #c_ie
    2.0000e+00, #c_ii
    4.0000e+00, #tau_e
    4.0000e+00, #tau_i
    0.0000e+00, #a_e
    1.4000e+00, #b_e
    1.0000e+00, #c_e
    0.0000e+00, #theta_e
    0.0000e+00, #a_i
    2.0000e+00, #b_i
    1.0000e+00, #c_i
    0.0000e+00, #theta_i
    0.5000e+00, #r_e
    0.5000e+00, #r_i
    0.5000e+00, #k_e
    0.0000e+00, #k_i
    0.0000e+00, #P
    0.0000e+00, #Q
    0.0000e+00, #alpha_e
    0.0000e+00, #alpha_i
    5.0000e-06, #Noise E
    5.0000e-06  #Noise I 
    ])
parameters_upper_bound_narrow = np.array([
    1.6000e+01, #c_ee
    1.5000e+01, #c_ei
    2.2000e+01, #c_ie
    1.5000e+01, #c_ii
    7.5000e+01, #tau_e
    7.5000e+01, #tau_i
    1.4000e+00, #a_e
    6.0000e+00, #b_e
    2.5000e+00, #c_e
    0.2500e+01, #theta_e
    2.0000e+00, #a_i
    6.0000e+00, #b_i
    2.5000e+00, #c_i
    0.25000e+01, #theta_i
    2.0000e+00, #r_e
    2.0000e+00, #r_i
    2.0000e+00, #k_e
    2.0000e+00, #k_i
    0.5000e+01, #P
    0.5000e+01, #Q
    0.5000e+01, #alpha_e
    0.5000e+01, #alpha_i
    1.0000e-04, #Noise E
    1.0000e-04  #Noise I 
    ])

parameters_range_bounds_narrow = parameters_upper_bound_narrow-parameters_lower_bound_narrow



parameters_lower_bound = np.array([
    1.1000e+01, #c_ee
    0.2000e+01, #c_ei
    0.2000e+01, #c_ie
    2.0000e+00, #c_ii
    4.0000e+00, #tau_e
    4.0000e+00, #tau_i
    0.0000e+00, #a_e
    1.4000e+00, #b_e
    1.0000e+00, #c_e
    0.0000e+00, #theta_e
    0.0000e+00, #a_i
    2.0000e+00, #b_i
    1.0000e+00, #c_i
    0.0000e+00, #theta_i
    0.5000e+00, #r_e
    0.5000e+00, #r_i
    0.5000e+00, #k_e
    0.0000e+00, #k_i
    0.0000e+00, #P
    0.0000e+00, #Q
    0.0000e+00, #alpha_e
    0.0000e+00, #alpha_i
    5.0000e-06, #Noise E
    5.0000e-06  #Noise I 
    ])
parameters_upper_bound = np.array([
    1.6000e+01, #c_ee
    1.5000e+01, #c_ei
    2.2000e+01, #c_ie
    1.5000e+01, #c_ii
    7.5000e+01, #tau_e
    7.5000e+01, #tau_i
    1.4000e+00, #a_e
    6.0000e+00, #b_e
    5.0000e+00, #c_e
    1.0000e+01, #theta_e
    2.0000e+00, #a_i
    6.0000e+00, #b_i
    5.0000e+00, #c_i
    1.0000e+01, #theta_i
    2.0000e+00, #r_e
    2.0000e+00, #r_i
    2.0000e+00, #k_e
    2.0000e+00, #k_i
    1.0000e+01, #P
    1.0000e+01, #Q
    1.0000e+01, #alpha_e
    1.0000e+01, #alpha_i
    1.0000e-04, #Noise E
    1.0000e-04  #Noise I 
    ])

parameters_range_bounds = parameters_upper_bound-parameters_lower_bound

parameters_alpha_peak_old_noise_implementation = np.array([1.6000e+01, 1.2000e+01, 1.5000e+01, 3.0000e+00, 28.0000e+00, 28.0000e+00,
        1.3000e+00, 4.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00, 3.7000e+00,
        1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.500e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,5.0000e-07,5.0000e-07])

parameters_lower_bound_old = np.array([
    1.1000e+01, #c_ee
    0.2000e+01, #c_ei
    0.2000e+01, #c_ie
    2.0000e+00, #c_ii
    3.0000e+00, #tau_e
    3.0000e+00, #tau_i
    0.0000e+00, #a_e
    1.4000e+00, #b_e
    1.0000e+00, #c_e
    0.0000e+00, #theta_e
    0.0000e+00, #a_i
    2.0000e+00, #b_i
    1.0000e+00, #c_i
    0.0000e+00, #theta_i
    0.5000e+00, #r_e
    0.5000e+00, #r_i
    0.5000e+00, #k_e
    0.0000e+00, #k_i
    0.0000e+00, #P
    0.0000e+00, #Q
    0.0000e+00, #alpha_e
    0.0000e+00, #alpha_i
    5.0000e-06, #Noise E
    5.0000e-06  #Noise I 
    ])


parameters_upper_bound_old = np.array([
    1.6000e+01, #c_ee
    1.5000e+01, #c_ei
    2.2000e+01, #c_ie
    1.5000e+01, #c_ii
    0.5000e+02, #tau_e
    0.7500e+02, #tau_i
    1.4000e+00, #a_e
    6.0000e+00, #b_e
    0.500e+01, #c_e
    1.0000e+01, #theta_e
    2.0000e+00, #a_i
    6.0000e+00, #b_i
    0.500e+01, #c_i
    1.0000e+01, #theta_i
    2.0000e+00, #r_e
    2.0000e+00, #r_i
    2.0000e+00, #k_e
    2.0000e+00, #k_i
    1.0000e+01, #P
    1.0000e+01, #Q
    1.0000e+01, #alpha_e
    1.0000e+01, #alpha_i
    1.0000e-04, #Noise E
    1.0000e-04  #Noise I 
    ])
parameters_range_bounds_old = parameters_upper_bound_old-parameters_lower_bound_old
