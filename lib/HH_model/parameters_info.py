from numpy import array as narray
from numpy import abs as nabs
from torch import tensor
worst_20 = tensor([413, 439, 435,  39, 195, 479, 499,  37, 398, 219,  35, 414, 476,  34,
         38, 255, 399, 474, 218, 235])

parameter_names = [
    'C', #in uF/cm2               0| 1.0
    'V_t',# in mV               1| -60
    'g_bar_m',# in mS/cm2        2| 0.07
    'g_k',# in mS/cm2             3| not fixed
    'E_k',# in mV              4| -107
    'g_na',# in mS/cm2           5| not fixed
    'E_na',# in mV               6| 53
    'g_l',# in mS/cm2             7| 0.1
    'E_leak',# in mV            8| -70
    'tau_max',# in ms             9| 6e2
    'noise',# in uA/cm2          10| 0.1
    'Current',# in nA           11| ?
]
parameters_initial0 = narray([
    1.0, #C in uF/cm2               0| 1.0
    -60.0, #V_t in mV               1| -60
    0.07, #g_bar_m in mS/cm2        2| 0.07
    1.0, #g_k in mS/cm2             3| not fixed
    -107.0, #E_k in mV              4| -107
    50.0, #g_na in mS/cm2           5| not fixed
    53.0, #E_na in mV               6| 53
    0.1, #g_l in mS/cm2             7| 0.1
    -70.0, #E_leak in mV            8| -70
    6e2, #tau_max in ms             9| 6e2
    0.1, #noise in uA/cm2          10| 0.1
    5e-4, #Current in nA           11| ?
    ])

parameters_initial1 = narray([
    1.0, #C in uF/cm2               0| 1.0
    -60.0, #V_t in mV               1| -60
    0.07, #g_bar_m in mS/cm2        2| 0.07
    1.5, #g_k in mS/cm2             3| not fixed
    -107.0, #E_k in mV              4| -107
    4.0, #g_na in mS/cm2            5| not fixed
    53.0, #E_na in mV               6| 53
    0.1, #g_l in mS/cm2             7| 0.1
    -70.0, #E_leak in mV            8| -70
    6e2, #tau_max in ms             9| 6e2
    0.1, #noise in uA/cm2          10| 0.1
    5e-4, #Current in nA           11| ?
    ])

parameters_initial2 = narray([
    1.0, #C in uF/cm2               0| 1.0
    -60.0, #V_t in mV               1| -60
    0.07, #g_bar_m in mS/cm2        2| 0.07
    15.0, #g_k in mS/cm2            3| not fixed
    -107.0, #E_k in mV              4| -107
    20.0, #g_na in mS/cm2           5| not fixed
    53.0, #E_na in mV               6| 53
    0.1, #g_l in mS/cm2             7| 0.1
    -70.0, #E_leak in mV            8| -70
    6e2, #tau_max in ms             9| 6e2
    0.1, #noise in uA/cm2          10| 0.1
    5e-4, #Current in nA           11| ?
    ])

parameters_lower_bound = narray([
    1.0000e-02, #C in uF/cm2        0| 1.0
    -9.000e+01, #V_t in mV          1| -60
    0.0000e+00, #g_bar_m in mS/cm2  2| 0.07
    0.0000e+00, #g_k in mS/cm2      3| not fixed
    -1.200e+02, #E_k in mV          4| -107
    0.0000e+00, #g_na in mS/cm2     5| not fixed
    2.0000e+01, #E_na in mV         6| 53
    0.0000e+00, #g_l in mS/cm2      7| 0.1
    -9.000e+01, #E_leak in mV       8| -70
    1.0000e+00, #tau_max in ms      9| 6e2
    1.0000e-02, #noise in uA/cm2   10| 0.1
    1.000e-05, #Current in nA     11| ?

    
    ])

parameters_upper_bound = narray([
    1.0000e+01, #C in uF/cm2        0| 1.0
    -4.000e+01, #V_t in mV          1| -60
    1.0000e+00, #g_bar_m in mS/cm2  2| 0.07
    1.5000e+02, #g_k in mS/cm2      3| not fixed
    -8.000e+01, #E_k in mV          4| -107
    1.5000e+02, #g_na in mS/cm2     5| not fixed
    7.0000e+01, #E_na in mV         6| 53
    1.0000e+00, #g_l in mS/cm2      7| 0.1
    -5.000e+01, #E_leak in mV       8| -70
    5.0000e+03, #tau_max in ms      9| 6e2
    1.0000e+00, #noise in uA/cm2   10| 0.1
    1.0000e-02, #Current in nA     11| ?

    
    ])
parameter_set_sbi = narray([parameters_initial0,parameters_initial1,parameters_initial2])
parameters_range_bounds = nabs(parameters_upper_bound-parameters_lower_bound)