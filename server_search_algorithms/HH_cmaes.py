from cma import CMAEvolutionStrategy
from joblib import Parallel, delayed
from os import makedirs
from torch import cat as tcat
from numpy.random import randint as nprandint
from datetime import datetime
from lib.Wilson_Cowan.search_utils import *
from lib.HH_model.parameters_info import worst_20, parameters_initial0, parameters_range_bounds, parameters_lower_bound, parameters_upper_bound
from lib.HH_model.simulator import HH_simulator
from copy import deepcopy as dcp

cos = CosineSimilarity(dim=1, eps=1e-8)
_, _ = HH_simulator(parameters_initial0.reshape((1,-1)), length = 0.01, dt = 0.01)
upper_bound = ensure_torch(parameters_upper_bound)
lower_bound = ensure_torch(parameters_lower_bound)
theta_min = ensure_torch(parameters_lower_bound)
theta_range = ensure_torch(parameters_range_bounds)
mid_point = ensure_torch(parameters_lower_bound + parameters_range_bounds/2)
#####################################################################--Define the search settings--#####################################################################
n_jobs = 30
save_file_name = 'cmaes_25chk_100rd_all_best_final25repeats'
targets_file = 'trial_1'
global_search_path = '../Data/HH/search_tests/cmaes/'
makedirs(global_search_path, exist_ok = True)
loss_fn = 'mse'
init_path ='../Data/HH/initialization/'
init_files = [
    'cube10000_0', 
    'cube10000_1',
]
script_id = 0

targets = []

for init_file in init_files:
    init_file_name = init_file
    target_log = get_log(init_path,init_file_name+'_fits')
    target_data = target_log['original']['fits']
    target_indices = worst_20
    target_temp = [[[target_data['target'][i], i],[target_data['theta'][i], target_data['x'][i], target_data['error'][i]]] for i in target_indices]
    #target_temp = [[[target_data['target'][i], i],[mid_point,None,10000000]] for i in target_indices]
    
    targets.append(target_temp)

queue = [[ii, i, std0] for i in range(20) for ii in range(len(init_files)) for std0 in [0.005]]
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
    simulated_samples, _ = HH_simulator(
                                                                        point,
                                                                        length = 0.05,
                                                        dt=0.01,
                                                        noise_seed=nprandint(0,2**16)
                                                )
    error = mse(target.squeeze(), simulated_samples)
    return [error.data.item(), [error.view(1,-1), simulated_samples.view(1,-1), point.view(1,-1)]]
test_fun = fun(norm_pars(ensure_torch(targets[0][0][1][0])).tolist())
def search_HH(x0, sigma0, target, initial_sol, max_iter, **kwargs):
    es = CMAEvolutionStrategy(x0, sigma0, {'bounds': [[0 for _ in range(12)],[1 for _ in range(12)]],'verbose':-1})
    sample_counter = 0
    search_log_info = {}
    search_log_info['target'] = target
    target = target[0].view(1,-1)
    # Keep track of the overall best
    overall_best_par = ensure_torch(initial_sol[0])
    #overall_best_x = ensure_torch(initial_sol[1])
    overall_best_error = ensure_torch(initial_sol[2])
    best_par_history = [ensure_torch(initial_sol[0]).view(1,-1)]
    #best_x_history = [ensure_torch(initial_sol[1]).view(1,-1)]
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
            #overall_best_x = round_best_x 
            overall_best_par = round_best_par
            overall_best_error = round_best_error
        # Store the round log into the search log
        #best_x_history.append(dcp(overall_best_x).view(1,-1))
        best_par_history.append(dcp(ensure_torch(overall_best_par)).view(1,-1))
        best_error_history.append(dcp(overall_best_error).view(1,-1))
        #es.logger.add()


    overall_runtime = datetime.now()-overall_st_time
    search_log_info["overall_runtime"] = ensure_torch(as_tensor([[overall_runtime.total_seconds()]]))
    search_log_info['best_fit'] = {
        #'x':overall_best_x.view(1,-1),
        'theta':overall_best_par.view(1,-1),
        'error':overall_best_error.view(1,-1)
    }
    search_log_info['fit_history'] = {
        #'x':tcat(best_x_history, dim = 0).unsqueeze(0),
        'theta':tcat(best_par_history, dim = 0).unsqueeze(0),
        'error':tcat(best_error_history, dim = 0).unsqueeze(0)
    }
    return search_log_info
        
def multi_restart_WC_search(num_restarts = 1,**kwargs):
    search_log = []
    for i in range(num_restarts):
        search_log.append(search_HH(**kwargs)) 
    search_template = {}
    search_template['settings'] = dcp(kwargs)
    search_template['target'] = [tcat([s['target'][0].unsqueeze(0) for s in search_log]).unsqueeze(0),tcat([s['target'][1].unsqueeze(0)for s in search_log]).unsqueeze(0)]
    search_template["overall_runtime"] = tcat([s['overall_runtime'] for s in search_log]).unsqueeze(0)
    search_template['best_fit'] = {
        #'x':tcat([s['best_fit']['x'] for s in search_log]).unsqueeze(0),
        'theta':tcat([s['best_fit']['theta'] for s in search_log]).unsqueeze(0),
        'error':tcat([s['best_fit']['error'] for s in search_log]).unsqueeze(0),
    }
    search_template['fit_history'] = {
        #'x':tcat([s['fit_history']['x'] for s in search_log]).unsqueeze(0),
        'theta':tcat([s['fit_history']['theta'] for s in search_log]).unsqueeze(0),
        'error':tcat([s['fit_history']['error'] for s in search_log]).unsqueeze(0),
    }
    return search_template

search_logs = Parallel(
                                        n_jobs=n_jobs,
                                        timeout = 999999999,
                                        verbose = 10
        )(delayed(multi_restart_WC_search)(
                                                    num_restarts = 25,
                                                    **search_settings,
                                                            
        ) for search_settings in list_search_settings)
save_log(search_logs, global_search_path, save_file_name)