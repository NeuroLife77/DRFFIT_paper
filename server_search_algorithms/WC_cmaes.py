from cma import CMAEvolutionStrategy
from joblib import Parallel, delayed
#from joblib.externals.loky import get_reusable_executor
from os import makedirs
from torch import cat as tcat
from numpy import array as narray
from numpy.random import randint as nprandint
from torch import logical_not as tlnot
from torch import logical_and as tland
from datetime import datetime
from lib.Wilson_Cowan.search_utils import *
from lib.Wilson_Cowan.parameters_info import avg_worst40, parameters_alpha_peak, parameters_range_bounds, parameters_lower_bound,parameters_upper_bound#, parameter_names
from lib.Wilson_Cowan.simulators import WC_stochastic_heun_PSD
from copy import deepcopy as dcp

cos = CosineSimilarity(dim=1, eps=1e-6)
def clean_PSD(simulated_samples, parameters_samples_good, filter_cutoff = [1e-6,1.0], frequency_range = [None,None]):
    simulated_samples = ensure_torch(simulated_samples)
    parameters_samples_good = ensure_torch(parameters_samples_good)
    not_nan = tlnot(simulated_samples.isnan().any(1))
    good_sim_low = simulated_samples[:, frequency_range[0]:frequency_range[1]].mean(1) > filter_cutoff[0]
    good_sim_high = simulated_samples[:, frequency_range[0]:frequency_range[1]].amax(1) < filter_cutoff[1]
    good_samples_scale = tland(good_sim_low,good_sim_high)
    good_sim = tland(not_nan,good_samples_scale)
    simulated_samples = simulated_samples[good_sim]
    parameters_samples_good = parameters_samples_good[good_sim]
    return simulated_samples, parameters_samples_good
upper_bound = ensure_torch(parameters_upper_bound)
lower_bound = ensure_torch(parameters_lower_bound)
theta_min = ensure_torch(parameters_lower_bound)
theta_range = ensure_torch(parameters_range_bounds)
mid_point = ensure_torch(parameters_lower_bound + parameters_range_bounds/2)
_, _, _ = WC_stochastic_heun_PSD(narray([parameters_alpha_peak]),length = 6, dt=1, get_psd_I = False, remove_bad=True)
frequency_range = [4,160]
cutoff = [0,200]
global_search_path = '../Data/WC/search_tests/cmaes/'
makedirs(global_search_path, exist_ok = True)
loss_fn = 'correlation'
init_path ='../Data/WC/initialization/'
init_files = [
    'cube_10000_0',
    'cube_10000_1',
]    
loss_fn = 'correlation'
targets = []

save_file_name = 'cmaes_25chk_100rd_all_best_final25repeats_0'

for init_file in init_files:
    init_file_name = init_file
    target_log = get_log(init_path,init_file_name+'_fits')
    target_data = target_log['original']['fits']
    target_indices = avg_worst40[:20]
    targets_one = [[[target_data['target'][i], i],[target_data['theta'][i], target_data['x'][i], target_data['error'][i]]] for i in target_indices]
    
    targets.append(targets_one)

queue = [[ii, i, std0] for i in range(20) for ii in range(len(init_files)-1) for std0 in [0.05]]
for i in range(20):
    for ii in range(1,2):
        for std0 in [0.025]:
            queue.append([ii, i, std0])
def norm_pars(pars):
    return (pars - theta_min) / theta_range
def denorm_pars(pars):
    return (pars * theta_range) + theta_min
list_search_settings = []
for ii, i, std0 in queue:
    max_iter = 2500
    x0 = norm_pars(ensure_torch(targets[ii][i][1][0])).numpy()
    initial_sol = ensure_torch(targets[ii][i][1])
    target = targets[ii][i][0]
    sigma0 = std0
    search_settings = {
        'x0': x0,
        'sigma0': sigma0,
        'target': target,
        'initial_sol': initial_sol,
        'max_iter': max_iter,
        'init_file':init_files[ii],
    }
    list_search_settings.append(search_settings)


def fun(theta, target = targets[0][0][0][0].view(-1,1)):
    theta_arr = ensure_torch(as_tensor(theta))
    point = denorm_pars(theta_arr).view(1,-1)
    simulated_samples, _, parameters_samples_good = WC_stochastic_heun_PSD(
                                                                        point,
                                                                        length = 302,
                                                                        dt=1.,
                                                                        noise_seed=nprandint(0,2**16),
                                                                        PSD_cutoff=frequency_range,
                                                                        remove_bad=True
                                                )
    simulated_samples, parameters_samples_good = clean_PSD(simulated_samples, parameters_samples_good, filter_cutoff = [1e-6,1.00])
    if simulated_samples is None or simulated_samples.shape[0] == 0:
        return [1., [1., None, None]]
    error = correlation_loss_fn(target, simulated_samples)
    return [error.data.item(), [error.view(1,-1), simulated_samples.view(1,-1), parameters_samples_good.view(1,-1)]]
test_fun = fun(norm_pars(ensure_torch(targets[0][0][1][0])).tolist())
def search_WC(x0, sigma0, target, initial_sol, max_iter, **kwargs):
    es = CMAEvolutionStrategy(x0, sigma0, {'bounds': [[0 for _ in range(24)],[1 for _ in range(24)]],'verbose':-1})
    sample_counter = 0
    search_log_info = {}
    search_log_info['target'] = target
    target = target[0].view(1,-1)
    # Keep track of the overall best
    overall_best_par = ensure_torch(initial_sol[0])
    overall_best_x = ensure_torch(initial_sol[1])
    overall_best_error = ensure_torch(initial_sol[2])
    best_par_history = [ensure_torch(initial_sol[0]).view(1,-1)]
    best_x_history = [ensure_torch(initial_sol[1]).view(1,-1)]
    best_error_history = [ensure_torch(initial_sol[2]).view(1,-1)]

    overall_runtime = datetime.now()-datetime.now()
    overall_st_time = datetime.now()
    errors = []
    while sample_counter < max_iter:
        X = es.ask(25)
        sample_counter += (len(X))
        res_er = []
        res_fit = []
        for x in X:
            err, fits = fun(x, target = target)
            res_er.append(err)
            res_fit.append(fits)
        res_fit.sort(key = lambda it: it[0])
        
        
        es.tell(X, res_er)

        round_best_x = res_fit[0][1]
        round_best_par = res_fit[0][2]
        round_best_error = res_fit[0][0]
        if round_best_error < overall_best_error:
            overall_best_x = round_best_x 
            overall_best_par = round_best_par
            overall_best_error = round_best_error
        # Store the round log into the search log
        best_x_history.append(dcp(overall_best_x).view(1,-1))
        best_par_history.append(dcp(ensure_torch(overall_best_par)).view(1,-1))
        best_error_history.append(dcp(overall_best_error).view(1,-1))
        es.logger.add()


    overall_runtime = datetime.now()-overall_st_time
    search_log_info["overall_runtime"] = ensure_torch(as_tensor([[overall_runtime.total_seconds()]]))
    search_log_info['best_fit'] = {
        'x':overall_best_x.view(1,-1),
        'theta':overall_best_par.view(1,-1),
        'error':overall_best_error.view(1,-1)
    }
    search_log_info['fit_history'] = {
        'x':tcat(best_x_history, dim = 0).unsqueeze(0),
        'theta':tcat(best_par_history, dim = 0).unsqueeze(0),
        'error':tcat(best_error_history, dim = 0).unsqueeze(0)
    }
    return search_log_info
        
def multi_restart_WC_search(num_restarts = 1,**kwargs):
    search_log = []
    for i in range(num_restarts):
        search_log.append(search_WC(**kwargs)) 
    search_template = {}
    search_template['settings'] = dcp(kwargs)
    search_template['target'] = [tcat([s['target'][0].unsqueeze(0) for s in search_log]).unsqueeze(0),tcat([s['target'][1].unsqueeze(0)for s in search_log]).unsqueeze(0)]
    search_template["overall_runtime"] = tcat([s['overall_runtime'] for s in search_log]).unsqueeze(0)
    search_template['best_fit'] = {
        'x':tcat([s['best_fit']['x'] for s in search_log]).unsqueeze(0),
        'theta':tcat([s['best_fit']['theta'] for s in search_log]).unsqueeze(0),
        'error':tcat([s['best_fit']['error'] for s in search_log]).unsqueeze(0),
    }
    search_template['fit_history'] = {
        'x':tcat([s['fit_history']['x'] for s in search_log]).unsqueeze(0),
        'theta':tcat([s['fit_history']['theta'] for s in search_log]).unsqueeze(0),
        'error':tcat([s['fit_history']['error'] for s in search_log]).unsqueeze(0),
    }
    return search_template

search_logs = Parallel(
                                        n_jobs=30,
                                        timeout = 999999999,
                                        verbose = 10
        )(delayed(multi_restart_WC_search)(
                                                    num_restarts = 25,
                                                    **search_settings,
                                                            
        ) for search_settings in list_search_settings)
save_log(search_logs, global_search_path, save_file_name)