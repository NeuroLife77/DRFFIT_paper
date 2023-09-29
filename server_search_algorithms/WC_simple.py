from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from os import makedirs
from torch import argmin as targmin
from torch import cat as tcat
from numpy import array as narray
from numpy import exp as npexp
from numpy.random import rand as nprand
from numpy.random import randint as nprandint
from torch import logical_not as tlnot
from torch import logical_and as tland
from datetime import datetime
from lib.Wilson_Cowan.search_utils import *
from lib.Wilson_Cowan.parameters_info import parameters_alpha_peak, parameters_range_bounds, parameters_lower_bound,parameters_upper_bound, avg_worst40
from lib.Wilson_Cowan.simulators import WC_stochastic_heun_PSD
from lib.drffit.uniform_sampler import uniform_around_sampler as uniform_sampler
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
cutoff = [0,200] # 100Hz (resolution of 0.5Hz)
#####################################################################--Define the search settings--#####################################################################
n_jobs = 20
global_search_path = '../Data/WC/search_tests/simple/'
makedirs(global_search_path, exist_ok = True)
loss_fn = 'correlation'
init_path ='../Data/WC/initialization/'

script_id = 0
file_tests = []

search_settings_list = [
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.0}, 'baseline': {'dist': 'sphere', 'width': 0.125}}, 'bas_dim': 0, 'init_file': 'cube_10000_0', 'LD': '5LD', 'tests': ['pure']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.0}, 'baseline': {'dist': 'sphere', 'width': 0.1}}, 'bas_dim': 0, 'init_file': 'cube_10000_0', 'LD': '5LD', 'tests': ['SA']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 8, 'init_file': 'cube_10000_0', 'LD': '5LD', 'tests': ['pure_AE']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 10, 'init_file': 'cube_10000_0', 'LD': '5LD', 'tests': ['pure_PCA']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 11, 'init_file': 'cube_10000_0', 'LD': '5LD', 'tests': ['SA_AE']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 10, 'init_file': 'cube_10000_0', 'LD': '5LD', 'tests': ['SA_PCA']} ,
    
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.0}, 'baseline': {'dist': 'sphere', 'width': 0.1}}, 'bas_dim': 0, 'init_file': 'cube_10000_1', 'LD': '5LD', 'tests': ['pure']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.0}, 'baseline': {'dist': 'sphere', 'width': 0.125}}, 'bas_dim': 0, 'init_file': 'cube_10000_1', 'LD': '5LD', 'tests': ['SA']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 8, 'init_file': 'cube_10000_1', 'LD': '5LD', 'tests': ['pure_AE']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 10, 'init_file': 'cube_10000_1', 'LD': '5LD', 'tests': ['pure_PCA']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.175}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 11, 'init_file': 'cube_10000_1', 'LD': '5LD', 'tests': ['SA_AE']} ,
        {'samp': {'drffit': {'dist': 'sphere', 'width': 0.15}, 'baseline': {'dist': 'sphere', 'width': 0.0}}, 'bas_dim': 12, 'init_file': 'cube_10000_1', 'LD': '5LD', 'tests': ['SA_PCA']} ,

]
for search_settings in search_settings_list:
    init_file_name = search_settings['init_file']
    LD = search_settings["LD"]
    drffit_file = init_file_name+'_DRFFIT_object'+LD
    target_logs = [get_log(init_path,init_file_name+'_fits')]
    sim_logs = [get_log(init_path,init_file_name)]
    loss_fn = 'correlation'
    drffit_path = '../Data/WC/initialization/DRFFIT_objects/'
    drffit_file = init_file_name+'_DRFFIT_object'+LD
    target_log = get_log(init_path,init_file_name+'_fits')
    target_data = target_log['original']['fits']
    target_indices = avg_worst40[:20]
    targets = [[[target_data['target'][i], i],[target_data['theta'][i], target_data['x'][i], target_data['error'][i]]] for i in target_indices]
    num_restarts = 25
    chunk_size = 25
    sample_distribution = search_settings['samp']['baseline']['dist']
    bl_width = search_settings['samp']['baseline']['width']
    search_width = search_settings['samp']['drffit']['width']
    drffit_sample_distribution = search_settings['samp']['drffit']['dist']
    num_basis_dim = search_settings['bas_dim']
    num_rounds = 100
    search_algs_selected = [test for test in search_settings['tests']]
    search_file_name = init_file_name+f'-{len(target_indices)}targets_w{search_width}_blw{bl_width}_chunk{chunk_size}_{num_rounds}rds_{num_basis_dim}dim_'+LD+'_'+search_algs_selected[0]+''
        
    temp = [2.5,2.5]
    num_dsamples = 10000
    
    statt = datetime.now()

    ##########################################################--Search function wrapper for parallel settings--#############################################################
    def search_WC(
                    target,
                    chunk_size = 10,
                    num_rounds = 5,
                    search_width = 0.1,
                    length = 302,
                    dt = 1.,
                    cutoff = [4,160],
                    loss_fn = correlation_loss_fn,
                    initial_sol = [mid_point, None, ensure_torch(as_tensor([1.]))],
                    temp = None,
                    augmented_search="none",
                    DRFFIT = None,
                    DRFFIT_info = None,
                    **kwargs
                ):
                
        # Keep track of the overall runtime
        overall_runtime = datetime.now()-datetime.now()
        overall_st_time = datetime.now()
        if temp is None:
            temp = 1e-1 * num_rounds
        # precompile simulator
        _, _, _ = WC_stochastic_heun_PSD(narray([parameters_alpha_peak]),length = 6, dt=1, get_psd_I = False, remove_bad=True)

    # Set an initial point to start the search around
        initial_point = initial_sol[0]#initial_sampler.sample((1,))
        
        # Initialize log of the search
        search_log_info = {}
        search_log_info['target'] = target#{'x':target, 'theta':target_par, 'real_index': }
        if initial_sol[1] is None:
            initial_sol[1] = ensure_torch(as_tensor([0 for _ in range(cutoff[0],cutoff[1])]))
        # Keep track of the overall best
        overall_best_par = initial_point
        overall_best_x = ensure_torch(initial_sol[1])
        overall_best_error = ensure_torch(initial_sol[2])

        # Keep track of the point around which the next search will be performed
        current_best_par = initial_point
        current_best_x = ensure_torch(initial_sol[1]).unsqueeze(0)
        current_best_error = ensure_torch(initial_sol[2]).unsqueeze(0)

        best_par_history = [ensure_torch(initial_sol[0]).unsqueeze(0)]
        best_x_history = [ensure_torch(initial_sol[1]).unsqueeze(0)]
        best_error_history = [ensure_torch(initial_sol[2]).unsqueeze(0)]
        
        for i in range(num_rounds):
            
            # Sampler settings
            point = current_best_par
            
            sampler = uniform_sampler(parameters_lower_bound, theta_range = parameters_range_bounds, sample_distribution = sample_distribution, **kwargs)
            
            # To keep track of round runtime
            runtime = datetime.now()-datetime.now()
            st_time = datetime.now()
            st_time_string = st_time.strftime('%D, %H:%M:%S')
            
            # Produce samples and simulate
            if DRFFIT is not None:
                if DRFFIT_info is None:
                    DRFFIT_info = {'SE_name': 'default','sampler_name': 'default','kwargs': {}}
                try:
                    dkwargs = DRFFIT_info['kwargs']
                except:
                    dkwargs = {}
                parameters_samples = DRFFIT.sample_in_subspace(
                                                                    point,
                                                                    search_width,
                                                                    chunk_size,
                                                                    sample_distribution = DRFFIT_info['sample_distribution'],
                                                                    min_norm = DRFFIT_info['min_norm'],
                                                                    subspace_estimator_name = DRFFIT_info['SE_name'],
                                                                    sampler_name = DRFFIT_info['sampler_name'],
                                                                    **dkwargs,**kwargs

                                                            )
            else:
                sampler.set_state(point = point, width=bl_width)
                parameters_samples = sampler.sample((chunk_size,))
            
            simulated_samples, _, parameters_samples_good = WC_stochastic_heun_PSD(
                                                                            parameters_samples,
                                                                            length = length,
                                                                            dt=dt,
                                                                            noise_seed=nprandint(0,2**16),
                                                                            PSD_cutoff=cutoff,
                                                                            remove_bad=True
                                                    )
                
            simulated_samples, parameters_samples_good = clean_PSD(simulated_samples, parameters_samples_good, filter_cutoff = [1e-6,1.00])
                
            # If no valid simulation remains
            if simulated_samples is None or simulated_samples.shape[0] == 0:
                if overall_best_error>1:
                    point = initial_point
                best_x_history.append(dcp(overall_best_x.unsqueeze(0)))
                best_par_history.append(dcp(ensure_torch(overall_best_par).unsqueeze(0)))
                best_error_history.append(dcp(overall_best_error).view(1,-1))
                continue
            
            # Define the log_info dict
            runtime = datetime.now()-st_time
            round_batch_error = loss_fn(target[0], simulated_samples)
            round_min_error = targmin(round_batch_error)
            
            # Evaluate best of the round
            round_best_x = simulated_samples[round_min_error]
            round_best_par = parameters_samples_good[round_min_error]
            round_best_error = round_batch_error[round_min_error]

            # Update the search best and overall best
            diff = overall_best_error-round_best_error
            t = temp / (i+1)
            metropolis = npexp(-diff / t)
            if round_best_error < current_best_error or ((nprand() < metropolis and augmented_search == "SA")):
                current_best_x = round_best_x 
                current_best_par = round_best_par
                current_best_error = round_best_error
            
            if current_best_error < overall_best_error:
                overall_best_x = current_best_x 
                overall_best_par = current_best_par
                overall_best_error = current_best_error
            
            # Store the round log into the search log
            best_x_history.append(dcp(overall_best_x.unsqueeze(0)))
            best_par_history.append(dcp(ensure_torch(overall_best_par).unsqueeze(0)))
            best_error_history.append(dcp(overall_best_error).view(1,-1))
            
            # Keep track of the progress
            f_time_string = datetime.now().strftime('%D, %H:%M:%S')
            runtime_string = str(runtime)
            parameters_samples = None
            simulated_samples = None
            round_best_x = None
            round_best_par = None
            round_best_error = None
            parameters_samples_good = None
            round_batch_error = None

        overall_runtime = datetime.now()-overall_st_time
        search_log_info["overall_runtime"] = ensure_torch(as_tensor([[overall_runtime.total_seconds()]]))
        search_log_info['best_fit'] = {
            'x':overall_best_x.unsqueeze(0),
            'theta':overall_best_par.unsqueeze(0),
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
        
    alg_indices = {'pure':0,'SA':1,'pure_AE':2,'pure_PCA':3, 'SA_AE':4, 'SA_PCA':5}    
    selected_alg = [alg_name in search_algs_selected for alg_name in alg_indices]    
    alg_options = [{
            'augmented_search':'none',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':bl_width,
            'sample_distribution_bl':sample_distribution,
        },{
            'augmented_search':'SA',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':bl_width,
            'sample_distribution_bl':sample_distribution,
            'temp': temp[0]
        },{
            'augmented_search':'none',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':search_width,
            'DRFFIT_info':{"SE_name":"default", "sampler_name":"default","sample_distribution":drffit_sample_distribution,
            "kwargs":{'num_eigen_vect_per_feature' : num_basis_dim, 'num_samples':num_dsamples}},
            'drffit': [drffit_path, drffit_file],
        },{
            'augmented_search':'none',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':search_width,
            'DRFFIT_info':{"SE_name":"PCA", "sampler_name":"PCA","sample_distribution":drffit_sample_distribution,
            "kwargs":{'num_eigen_vect_per_feature' : num_basis_dim, 'num_samples':num_dsamples}},
            'drffit': [drffit_path, drffit_file]
        },{
            'augmented_search':'SA',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':search_width,
            'temp': temp[1],
            'DRFFIT_info':{"SE_name":"default", "sampler_name":"default","sample_distribution":drffit_sample_distribution,
            "kwargs":{'num_eigen_vect_per_feature' : num_basis_dim, 'num_samples':num_dsamples}},
            'drffit': [drffit_path, drffit_file],
        },{
            'augmented_search':'SA',
            'num_rounds':num_rounds,
            'chunk_size':chunk_size,
            'search_width':search_width,
            'temp': temp[1],
            'DRFFIT_info':{"SE_name":"PCA", "sampler_name":"PCA","sample_distribution":drffit_sample_distribution,
            "kwargs":{'num_eigen_vect_per_feature' : num_basis_dim, 'num_samples':num_dsamples}},
            'drffit': [drffit_path, drffit_file]
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
        DRFFIT = None
        if 'drffit' in alg:
            DRFFIT = get_log(alg['drffit'][0], alg['drffit'][1])
        print('\n',alg, ':\n')
        search_log = Parallel(
                                        n_jobs=n_jobs,
                                        timeout = 999999999,
                                        verbose = 10
        )(delayed(multi_restart_WC_search)(
                                                    num_restarts = num_restarts,
                                                    target = target[0],
                                                    initial_sol = target[1],
                                                    DRFFIT = DRFFIT,
                                                    **alg,
                                                            
        ) for target in targets)
        
        search_template = {}
        search_template['settings'] = dcp(alg)
        search_template['target'] = [tcat([s['target'][0].unsqueeze(0) for s in search_log]),tcat([s['target'][1].unsqueeze(0) for s in search_log])]
        search_template["overall_runtime"] = tcat([s['overall_runtime'] for s in search_log])
        search_template['best_fit'] = {
            'x':tcat([s['best_fit']['x'] for s in search_log]),
            'theta':tcat([s['best_fit']['theta'] for s in search_log]),
            'error':tcat([s['best_fit']['error'] for s in search_log]),
        }
        search_template['fit_history'] = {
            'x':tcat([s['fit_history']['x'] for s in search_log]),
            'theta':tcat([s['fit_history']['theta'] for s in search_log]),
            'error':tcat([s['fit_history']['error'] for s in search_log]),
        }
        search_logs.append(search_template)
        #get_reusable_executor().shutdown(wait = True)
    makedirs(global_search_path, exist_ok = True)
    save_log(search_logs, global_search_path, search_file_name)
    file_tests.append(search_file_name)
    runtime = datetime.now()-statt
    print(str(runtime))
