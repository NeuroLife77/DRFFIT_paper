from joblib import Parallel, delayed
#from joblib.externals.loky import get_reusable_executor
from os import makedirs
from torch import argmin as targmin
from torch import cat as tcat
#from numpy import array as narray
from numpy import exp as npexp
from numpy.random import rand as nprand
from numpy.random import randint as nprandint
#from torch import logical_not as tlnot
#from torch import logical_and as tland
from torch import ones as tones
from datetime import datetime
from lib.Wilson_Cowan.search_utils import *
from lib.HH_model.parameters_info import worst_20,parameters_initial0, parameters_range_bounds, parameters_lower_bound, parameters_upper_bound
from lib.drffit.uniform_sampler import uniform_around_sampler as uniform_sampler
from lib.HH_model.simulator import HH_simulator
from copy import deepcopy as dcp
cos = CosineSimilarity(dim=1, eps=1e-8)
_, _ = HH_simulator(parameters_initial0.reshape((1,-1)), length = 0.01, dt = 0.01)
upper_bound = ensure_torch(parameters_upper_bound)
lower_bound = ensure_torch(parameters_lower_bound)
theta_min = ensure_torch(parameters_lower_bound)
theta_range = ensure_torch(parameters_range_bounds)
mid_point = ensure_torch(parameters_lower_bound + parameters_range_bounds/2)
def norm_pars(pars):
    return (pars - theta_min) / theta_range
def denorm_pars(pars):
    return (pars * theta_range) + theta_min
#####################################################################--Define the search settings--#####################################################################
n_jobs = 20
targets_file = 'trial_1'
global_search_path = '../Data/HH/'
makedirs(global_search_path, exist_ok = True)
loss_fn = 'mse'
init_path ='../Data/HH/initialization/'
glob_data_path =  '../Data/HH/search_tests/PSO/'
script_id = 0
search_settings_list = [
    {'samp': {'baseline': {'dist': 'sphere', 'width': 0.15}}, 'bas_dim': 0, 'init_file': 'cube10000_0', 'LD': '', 'tests': ['PSO'], 'pso_pars': [0.5, 1.5, 0.25]} ,
    {'samp': {'baseline': {'dist': 'sphere', 'width': 0.15 }}, 'bas_dim': 0, 'init_file': 'cube10000_1', 'LD': '', 'tests': ['PSO'], 'pso_pars': [0.5, 1.5, 0.5]} ,

] 
file_tests = []
for search_settings in search_settings_list:
    loss_fn = 'mse'
    pso_pars=search_settings['pso_pars']
    init_file_name = search_settings['init_file']
    target_log = get_log(init_path,init_file_name+'_fits')
    target_data = target_log['original']['fits']
    target_indices = worst_20
    targets = [[[target_data['target'][i], i],[target_data['theta'][i], target_data['x'][i], target_data['error'][i]]] for i in target_indices]
    num_restarts = 25
    chunk_size = 25
    sample_distribution = search_settings['samp']['baseline']['dist']
    bl_width = search_settings['samp']['baseline']['width']
    num_basis_dim = search_settings['bas_dim']
    num_rounds = 100
    search_algs_selected = [test for test in search_settings['tests']]
    search_file_name = init_file_name+f'-{len(target_indices)}targets_blw{bl_width}_chunk{chunk_size}_{num_rounds}rds_w{pso_pars[0]}_c{pso_pars[1]}_c{pso_pars[2]}_{script_id}_'+search_algs_selected[0]
        
    pso_pars = pso_pars
    
    statt = datetime.now()

    ##########################################################--Search function wrapper for parallel settings--#############################################################
    def search_HH(
                    target,
                    chunk_size = 10,
                    num_rounds = 5,
                    length = 0.05,
                    dt = 0.01,
                    loss_fn = mse,
                    initial_sol = [mid_point, None, ensure_torch(as_tensor([1.]))],
                    pso_pars = [0.9, 2., 2.],
                    **kwargs
                ):
                
        # Keep track of the overall runtime
        overall_runtime = datetime.now()-datetime.now()
        overall_st_time = datetime.now()
        # precompile simulator
        _, _ = HH_simulator(parameters_initial0.reshape((1,-1)), length = 0.01, dt = 0.01)
        # Set an initial point to start the search around
        initial_point = initial_sol[0]#initial_sampler.sample((1,))
        
        # Initialize log of the search
        search_log_info = {}
        search_log_info['target'] = target#{'x':target, 'theta':target_par, 'real_index': }
        #print(initial_sol[2])
        w, c_0, c_1 = pso_pars[0], pso_pars[1], pso_pars[2]
        # Keep track of the overall best
        overall_best_par = initial_point
        overall_best_x = ensure_torch(initial_sol[1])
        overall_best_error = ensure_torch(initial_sol[2])

        # Keep track of the point around which the next search will be performed
        current_best_par = initial_point
        #current_best_x = ensure_torch(initial_sol[1]).unsqueeze(0)
        #current_best_error = ensure_torch(initial_sol[2]).unsqueeze(0)

        best_par_history = [ensure_torch(initial_sol[0]).unsqueeze(0)]
        best_x_history = [ensure_torch(initial_sol[1]).unsqueeze(0)]
        best_error_history = [ensure_torch(initial_sol[2]).unsqueeze(0)]
        
        sampler = uniform_sampler(parameters_lower_bound, theta_range = parameters_range_bounds, sample_distribution = sample_distribution, **kwargs)
        sampler.set_state(point = current_best_par, width=bl_width)   
        particle_swarm = sampler.sample((chunk_size,))
        velocity = particle_swarm - sampler.sample((chunk_size,))
        particle_swarm[0] = current_best_par
        
        swarm_best_error = tones(chunk_size,1).view(-1)
        swarm_best = dcp(particle_swarm)
        for i in range(num_rounds):
            
            simulated_samples, _ = HH_simulator(
                                                    particle_swarm,
                                                    length = length,
                                                    dt=dt,
                                                    noise_seed=nprandint(0,2**16)
                                                )
                
                
            
            # Define the log_info dict
            # Define the log_info dict
            round_batch_error = mse(target[0], simulated_samples)
            improved_particles = round_batch_error<swarm_best_error.view(-1)
            swarm_best[improved_particles] = particle_swarm[improved_particles]
            swarm_best_error[improved_particles] = round_batch_error[improved_particles]
            
            
            round_min_error = targmin(round_batch_error)
            # Evaluate best of the round
            round_best_x = simulated_samples[round_min_error]
            round_best_par = particle_swarm[round_min_error]
            round_best_error = round_batch_error[round_min_error]
            if round_best_error < overall_best_error:
                overall_best_x = round_best_x 
                overall_best_par = round_best_par
                overall_best_error = round_best_error
            
            deviation_best = norm_pars(overall_best_par) - norm_pars(particle_swarm)
            deviation_swarm_best = norm_pars(swarm_best) - norm_pars(particle_swarm)
            velocity = w * velocity + c_0 * ensure_torch(nprand(chunk_size)).view(-1,1) * deviation_best + c_1 * ensure_torch(nprand(chunk_size)).view(-1,1) * deviation_swarm_best
            normalized_particles = norm_pars(particle_swarm) + velocity 
            particle_swarm = denorm_pars(normalized_particles)

            # Store the round log into the search log
            #best_x_history.append(dcp(overall_best_x.unsqueeze(0)))
            best_par_history.append(dcp(ensure_torch(overall_best_par).unsqueeze(0)))
            best_error_history.append(dcp(overall_best_error).view(1,-1))
            
            # Keep track of the progress
            simulated_samples = None
            round_best_x = None
            round_best_par = None
            round_best_error = None
            round_batch_error = None

        overall_runtime = datetime.now()-overall_st_time
        search_log_info["overall_runtime"] = ensure_torch(as_tensor([[overall_runtime.total_seconds()]]))
        search_log_info['best_fit'] = {
            #'x':overall_best_x.unsqueeze(0),
            'theta':overall_best_par.unsqueeze(0),
            'error':overall_best_error.view(1,-1)
        }
        search_log_info['fit_history'] = {
            #'x':tcat(best_x_history, dim = 0).unsqueeze(0),
            'theta':tcat(best_par_history, dim = 0).unsqueeze(0),
            'error':tcat(best_error_history, dim = 0).unsqueeze(0)
        }

        return search_log_info


    def multi_restart_HH_search(num_restarts = 1,**kwargs):
        search_log = []
        for i in range(num_restarts):
            search_log.append(search_HH(**kwargs)) 
        search_template = {}
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
        
    alg_indices = {'PSO':0} 
    selected_alg = [alg_name in search_algs_selected for alg_name in alg_indices]    
    alg_options = [{
            'augmented_search':'PSO',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':bl_width,
            'sample_distribution_bl':sample_distribution,
            'pso_pars':pso_pars,
        }
    ]
    print(selected_alg)
    algs = []
    for i, optn in enumerate(alg_options):
        if selected_alg[i]:
            algs.append(optn)
        else:
            algs.append(None)

    search_logs = []
    for ii, alg in enumerate(algs):
        if alg is None:
            search_logs.append(None)
            continue
        print('\n',alg, ':\n')
        search_log = Parallel(
                                        n_jobs=n_jobs,
                                        timeout = 999999999,
                                        verbose = 10
        )(delayed(multi_restart_HH_search)(
                                                    num_restarts = num_restarts,
                                                    target = target[0],
                                                    initial_sol = target[1],
                                                    **alg,
                                                            
        ) for target in targets)
        
        search_template = {}
        search_template['settings'] = dcp(alg)
        search_template['target'] = [tcat([s['target'][0].unsqueeze(0) for s in search_log]),tcat([s['target'][1].unsqueeze(0) for s in search_log])]
        search_template["overall_runtime"] = tcat([s['overall_runtime'] for s in search_log])
        search_template['best_fit'] = {
            #'x':tcat([s['best_fit']['x'] for s in search_log]),
            'theta':tcat([s['best_fit']['theta'] for s in search_log]),
            'error':tcat([s['best_fit']['error'] for s in search_log]),
        }
        search_template['fit_history'] = {
            #'x':tcat([s['fit_history']['x'] for s in search_log]),
            'theta':tcat([s['fit_history']['theta'] for s in search_log]),
            'error':tcat([s['fit_history']['error'] for s in search_log]),
        }
        search_logs.append(search_template)
    makedirs(global_search_path, exist_ok = True)
    save_log(search_logs, global_search_path, search_file_name)
    file_tests.append(search_file_name)
    runtime = datetime.now()-statt
    print(str(runtime))
