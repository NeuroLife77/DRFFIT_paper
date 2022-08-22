from lib.drffit.subspace_estimator import *
from lib.Feature_extraction.AE import AE as AutoEncoder
import warnings
import numpy as np
import torch
from lib.utils import ensure_torch, ensure_numpy

warnings.simplefilter("always")

class DRFFIT:
    def __init__(self,theta_dim, output_dim, theta_min, theta_range, default_verbose = 1, *args, **kwargs):
        """     This function allows the creation of a DRFFIT object
        Args:
            theta_dim:          The number of dimensions of the parameter space
            output_dim:         The output dimension of the simulator
            theta_min:          The lower bound for the parameters
            theta_range:        The range for the parameter search (equivalent to: theta_max-theta_min)
            default_verbose:    The default verbose state of the object
        """

        # Info on the simulator and parameter space
        self.theta_dim = theta_dim
        self.output_dim = output_dim
        self.theta_min = torch.as_tensor(theta_min).float()
        self.theta_range = torch.as_tensor(theta_range).float()

        # Dict of SE and name of the one currently set as default
        self.subspace_estimator = {}
        self.default_subspace_estimator_name = "first"
        self.sampler = {}

        # List of feature names and dict of feature functions
        self.feature_fn_name = []
        self.feature_fn = {}

        # Dict of the data stored in the DRFFIT object
        self.data = {}

        #VErbose state
        self.default_verbose = default_verbose
        

    def initialize_subspace_estimator(self, name = "default", replace = False, set_as_default = False, *args, **kwargs):
        """     This function allows the creation of a subspace estimator (SE)
        Args:
            name:           The name to give to the SE
            replace:        If an SE exists under that name, should it be replaced by the new SE
            set_as_default: If the SE is to be set as the default SE
        """
        
        try: #If it already exists
            _ = self.subspace_estimator[name]
            
            if not replace and not set_as_default:
                warnings.warn(f"'{name}' already exists. Select replace = True if you want to replace the existing one, specify a name to create a new one, and select set_as_default = True to make it the new default", stacklevel=2)
                return
            
            if set_as_default:
                if not replace and self.default_subspace_estimator_name == name:
                    warnings.warn(f"'{name}' is already default", stacklevel=2)
                    return
                # Set the new default SE and store its name
                self.subspace_estimator[self.default_subspace_estimator_name] = self.subspace_estimator["default"]
                self.subspace_estimator["default"] = self.subspace_estimator.pop(name)
                self.default_subspace_estimator_name = name

        except: #If it doesn't 
            # Edge case where it does exist and is already the 'default'
            if self.default_subspace_estimator_name == name:
                warnings.warn(f"'{name}' is already the default subspace_estimator. Select replace = True if you want to replace the existing one, specify a name to create a new one", stacklevel=2)
                return
            
            # Create new SE
            self.subspace_estimator[name] = {"subspace_estimator":subspace_estimator(self.theta_min, self.theta_range, input_size = self.theta_dim), "features":[]}
            self.sampler[name] = {}
            # To set the new SE as default if a specific name was given
            if set_as_default and name != "default":
                self.subspace_estimator[self.default_subspace_estimator_name] = self.subspace_estimator["default"]
                self.subspace_estimator["default"] = self.subspace_estimator.pop(name)
                self.default_subspace_estimator_name = name
            self.sampler[self.default_subspace_estimator_name] = {}


    def del_subspace_estimator(self, name, del_default = False, *args, **kwargs):
        """     This function allows the deletion of a subspace estimator (SE)
        Args:
            name:           The name of the SE
            del_default:    Safety argument to prevent deleting the default SE without explicit intent
        """
        
        if name == self.default_subspace_estimator_name or name == "default":
            if del_default:
                _ = self.subspace_estimator.pop("default")
                self.default_subspace_estimator_name = "-"
                warnings.warn(f"You just deleted the default subspace estimator, to continue create or set the new default using 'initialize_subspace_estimator'", stacklevel=2)
            else:
                # Safety process to avoid deleting default without explicit intent
                warnings.warn(f"This is the default subspace estimator, to delete it please select del_default = True", stacklevel=2)
                return
        try:
            _ = self.subspace_estimator.pop(name)
        except:
            warnings.warn(f"There was no subspace_estimator under the name '{name}'", stacklevel=2)


    def add_feature_fn_to_subspace_estimator(self, feature_fn_name = "default_AE", feature_index = 0, subspace_estimator_name = "default", custom_net = None, *args, **kwargs):
        """     This function adds a feature function to a subspace estimator (SE)
        Args:
            feature_fn_name:           The name of the feature function
            feature_index:             The index of the specific feature. To be specified if the feature function returns more than 1 feature 
            subspace_estimator_name:   The name of the SE
            custom_net:                Can take a custom pytorch neural network object (must have a function called 'feature_fn' that returns the feature values)
            **kwargs:                  Will be passed to the .add_feature() so it can take the arguments 'architecture' or 'num_units' to specify a specific architecture 
        """

        # Get the SE
        model = self.get_subspace_estimator(subspace_estimator_name)
        if model == None:
            return
        
        try:
            # Add the feature
            model["features"].append((feature_fn_name, feature_index))
        except:
            warnings.warn(f"'{subspace_estimator_name}' is an invalid subspace_estimator [it has no list of features]", stacklevel=2)
            return
        
        try:
            if custom_net == None:
                # Add the regression network for the corresponding feature
                model["subspace_estimator"].add_feature(**kwargs)
                return
            else:
                # Add the custom network for the corresponding feature
                model["subspace_estimator"].add_custom_net_feature(custom_net, **kwargs)
                return
        except:
            warnings.warn(f"'{subspace_estimator_name}' is an invalid subspace_estimator [it has no list of features]", stacklevel=2)
            return


    def set_feature_data_of_subspace_estimator(self, subspace_estimator_name = "default", data = "default", data_name = "default", store_feature_data = False, overwrite_data = False, rename_data = None, set_as_default = False, *args, **kwargs):
        """     This function attaches the feature data to a subspace estimator (SE)
        Args:
            subspace_estimator_name:   The name of the SE
            data:                      The name of the group of datasets
            data_name:                 The name of the dataset 
            store_feature_data:        Define if the feature values will be stored in the DRFFIT object after being computed (if they aren't already stored)
            overwrite_data:            Define if the feature values computed are to overwrite currently stored data in the SE object (if there)
            rename_data:               Define the new name of the data within the SE object if desired
            set_as_default:            Define if the data passed is to be set as the default data for the SE
            **kwargs:                  Can be used to specify the parameters of the .set_data() of the SE object
        """

        try: # If the group of datasets sexists
            _ = self.data[data]
        except:
            warnings.warn(f"'{data}' is not a known set of datasets. You can add a set of data with the 'add_data()' method ", stacklevel=2)
            return
        
        try: # If the dataset exists
            dataset = self.data[data][data_name]
        except:
            warnings.warn(f"{data_name} is not in '{data}'", stacklevel=2)
            return
        
        # Get the data
        simulations = dataset["x"]
        thetas = dataset["theta"]
        if thetas == None:
            warnings.warn(f"{data_name} of '{data}' has no associated parameter values. It cannot be used to train the subspace_estimator", stacklevel=2)
            return
        
        # Get the model
        model = self.get_subspace_estimator(subspace_estimator_name)
        if model == None:
            return
        
        # Set the name to be used in the SE object for the data
        subspace_estimator_data_name = rename_data
        if rename_data == None:
            subspace_estimator_data_name = data+" "+data_name
        
        try: # If the data already exists in the SE
            if not overwrite_data and not set_as_default:
                _ = model["subspace_estimator"].prev_data[subspace_estimator_data_name]
                warnings.warn(f"Failed to set data: '{data_name}' of '{data}' already exists within the subspace_estimator '{subspace_estimator_name}' under the name '{subspace_estimator_data_name}'. You can overwrite it with overwrite_data = True, add it with a new name using the 'rename_data' argument or make it the default for subspace_estimator '{subspace_estimator_name}' using 'set_as_default = True'", stacklevel=2)
                return
            
            if set_as_default:
                _ = subspace_estimator.set_prev_data(subspace_estimator_data_name)
                return
        except:
            pass
        
        # Get the feature values
        if dataset["features"] == None:

            # To be able to report which feature caused an error if there is one
            current_feature =  "no existing feature functions" 
            
            # Compute the feature values
            feature_values = []
            #try:
                # List all distinct feature functions
            feature_fns = []
            for f in model["features"]:
                if not f[0] in feature_fns:
                    feature_fns.append(f[0])

            # Compute the values for all the feature functions 
            feature_vals = {}
            for f in feature_fns:
                current_feature = f
                feature_fn = self.get_fn(f)
                feature = feature_fn.feature_fn(simulations)
                feature_vals[f] = feature

            # Keep only the values of the features specified in the SE
            for feature_fn_name, feature_index in model["features"]:
                feature = feature_vals[feature_fn_name][:,feature_index]
                feature_values.append(torch.as_tensor(feature).view(-1,1))
                
           # except:
           #     warnings.warn(f"Error with the feature function '{current_feature}' of '{subspace_estimator_name}' with dataset '{data_name}' in '{data}'", stacklevel=2)
           #     return
            
            # Group and store the feature values in the SE
            feature_values = torch.cat(feature_values, dim = 1)
            model["subspace_estimator"].set_data(thetas, feature_values, data_name = subspace_estimator_data_name, **kwargs)
            
            # Store the feature values in the DRFFIT object
            if store_feature_data:
                dataset["features"] = feature_values
        else:
            # Load the feature values from the DRFFIT object into the SE
            feature_values = dataset["features"]
            model["subspace_estimator"].set_data(thetas, feature_values, data_name = subspace_estimator_data_name, **kwargs)
    
    
    def train_net_from_subspace_estimator(self, feature_index, name = 'default', verbose = -1, enforce_replace = None, show_feature_verbose = True, data = None, *args, **kwargs):
        """     This function trains a specific regression network in a subspace estimator (SE)
        Args:
            feature_index:             The index of the feature to be trained
            name:                      The name of the SE object
            verbose:                   Allow a specified verbose state to overwrite the default
            enforce_replace:           Enforce a reset on the network parameters if the network has already been trained
            show_feature_verbose:      Prints the name of the feature in the training
            data:                      Specify a dataset in the SE object to train the network (else the SE's default is used)
            **kwargs:                  Can be used to specify the parameters of the .train() and .set_prev_data() of the SE object
        """

        # Get the SE   
        model = self.get_subspace_estimator(name)
        if model == None:
            return
        subspace_estimator = model["subspace_estimator"]

        # Handle missing data for training
        if not subspace_estimator.has_data:
            warnings.warn(f"'{name}' cannot be trained because it has no data. Please set the model's data first. Ex: You can use the 'set_feature_data_of_subspace_estimator()' method", stacklevel=2)
            return

        # Handle if it was already trained    
        if subspace_estimator.net_has_trained[feature_index]:
            if enforce_replace == None:
                warnings.warn(f"Failed to train '{name}' subspace_estimator: The subspace_estimator networks for that feature has already been trained.\n Pleace specify if you want to replace the current models or train from the current state using the 'enforce_replace' argument. Note that you can save the current version before training or replacing by calling the 'save_subspace_estimator()' function", stacklevel=2)
                return
            if enforce_replace:
                subspace_estimator.reset_feature(feature_index)
        
        # Use default verbose state
        if verbose == -1:
            verbose = self.default_verbose
        
        if data == None: # Train with SE's default data
            if show_feature_verbose:
                print(model['features'][feature_index])
            subspace_estimator.train(feature_index, verbose = verbose, **kwargs)
        else:
            try:
                # Set the specified data as default (temporary)
                curr_data = subspace_estimator.set_prev_data(data, **kwargs) # This returns the previous default's name
                if curr_data != None:
                    # Train the regression network
                    if show_feature_verbose:
                        print(model['features'][feature_index])
                    subspace_estimator.train(feature_index, verbose = verbose, **kwargs)
                    
                    # Reinstate the previous default data
                    _ = subspace_estimator.set_prev_data(curr_data)
                    return
                warnings.warn(f"'{name}' cannot be trained because data '{data}' does not exist. Please make sure that it exists or add it through the 'set_feature_data_of_subspace_estimator()' method before training", stacklevel=2)
                return
            except:
                warnings.warn(f"'{name}' cannot be trained because an error occured when setting data '{data}'. Please make sure that it has the correct arguments stored method before training", stacklevel=2)
                return
    
    
    def train_all_nets_from_subspace_estimator(self, name = 'default', verbose = -1, enforce_replace = None, skip_trained = True, show_feature_verbose = True, data = None, return_val = False, *args, **kwargs):
        """     This function trains all regression networks in a subspace estimator (SE)
        Args:
            feature_index:             The index of the feature to be trained
            name:                      The name of the SE object
            verbose:                   Allow a specified verbose state to overwrite the default
            enforce_replace:           Enforce a reset on the network parameters if the network has already been trained
            skip_trained:              Defines if the already trained networks are to be skipped or trained further from their current state
            show_feature_verbose:      Prints the name of the feature in the training
            data:                      Specify a dataset in the SE object to train the network (else the SE's default is used)
            **kwargs:                  Can be used to specify the parameters of the .train_all() and .set_prev_data() of the SE object
        """

        # Get the SE
        model = self.get_subspace_estimator(name)
        if model == None:
            return
        subspace_estimator = model["subspace_estimator"]

        # Handle missing data for training
        if not subspace_estimator.has_data:
            warnings.warn(f"'{name}' cannot be trained because it has no data. Please set the model's data first. Ex: You can use the 'set_feature_data_of_subspace_estimator()' method", stacklevel=2)
            return

        # Identify which networks have already been trained
        already_trained = []
        skip = []
        for i in range(len(subspace_estimator.net_has_trained)):
            if subspace_estimator.net_has_trained[i]:
                already_trained.append(model['features'][i])
                skip.append(i)
        
        # Deal with the networks that have already been trained
        if len(already_trained)>0:
            if enforce_replace == None:
                warnings.warn(f"Failed to train '{name}' subspace_estimator: The subspace_estimator networks for the following features have already been trained: {already_trained} \n Pleace specify how you want to resolve this using the 'enforce_replace' and 'skip_trained' arguments. Note that you can save the current version before training or replacing by calling the 'save_subspace_estimator()' function", stacklevel=2)
                return
            if not skip_trained:
                skip = []
            for i in range(len(subspace_estimator.net_list)):
                if subspace_estimator.net_has_trained[i] and enforce_replace:
                    subspace_estimator.reset_feature(i)
        
        # Set default verbose state
        if verbose == -1:
            verbose = self.default_verbose
        
        if data == None: # Train with SE's default data
            if show_feature_verbose:
                if return_val:
                    val = subspace_estimator.train_all(verbose = verbose, features = model['features'], skip = skip,return_val = return_val, **kwargs)
                    return val
                subspace_estimator.train_all(verbose = verbose, features = model['features'], skip = skip, **kwargs)
            else:
                if return_val:
                    val = subspace_estimator.train_all(verbose = verbose, skip = skip,return_val = return_val, **kwargs)
                    return val
                subspace_estimator.train_all(verbose = verbose, skip = skip, **kwargs)
        else: # Train with specific dataset
            try:
                # Set the specified data as default (temporary) 
                curr_data = subspace_estimator.set_prev_data(data, **kwargs) # This returns the previous default's name
                if curr_data != None:
                    # Train the regression networks
                    if show_feature_verbose:
                        subspace_estimator.train_all(verbose = verbose, features = model['features'], skip = skip, **kwargs)
                        curr_data = subspace_estimator.set_prev_data(curr_data)
                    else:
                        subspace_estimator.train_all(verbose = verbose, skip = skip, **kwargs)
                        curr_data = subspace_estimator.set_prev_data(curr_data)
                    return
                warnings.warn(f"'{name}' cannot be trained because data '{data}' does not exist. Please make sure that it exists or add it through the 'set_feature_data_of_subspace_estimator()' method before training", stacklevel=2)
                return
            except:
                warnings.warn(f"'{name}' cannot be trained because an error occured when setting data '{data}'. Please make sure that it has the correct arguments stored method before training", stacklevel=2)
                return


    def get_feature_active_subspace_from_subspace_estimator(self, feature_index, samples = None, num_samples = 10000, sample_from_default_sampler_state = True, name = 'default', sampler_name = 'default', num_eigen_vect_per_feature = 1, return_samples = False, *args, **kwargs):
        """     This function obtains basis vector(s) of the active subspace for specified feature using the regression network of the subspace estimator (SE)
        Args:
            feature_index:                          The index of the feature for which the basis vector(s)
            samples:                                Parameter samples used for estimation of the basis vector(s) of the active subspace (if none, they are taken from the sampler)
            num_samples:                            Specify the number of samples to take from the sampler if not samples are passed
            sample_from_default_sampler_state:      Define if the samples are to be taken from default state of the sampler
            name:                                   The name of the SE object
            sampler_name:                           The name of the sampler to be used it no samples are passed
            num_eigen_vect_per_feature:             Specify the number of eigenvectors of the active subspace for each feature to use in the DRFFIT subspace 
            return_samples:                         Define if the samples are to be returned along with the subspace
            **kwargs:                               Only to collect additional arguments (avoid crashing)
        """
        
        # Get the SE
        model = self.get_subspace_estimator(name)
        if model == None:
            return
        
        if name == 'default':
            name = self.default_subspace_estimator_name

        if samples == None:
            if not sample_from_default_sampler_state:
                try:
                    self.sampler[name][sampler_name].set_state(**kwargs)
                except:
                    try:
                        self.sampler[name][sampler_name].set_default_x(**kwargs)
                    except:
                        warnings.warn(f"Could not set the sampler state using 'set_state' nor 'set_default_x', using default sampler state.", stacklevel=2)
            try:
                samples = self.sampler[name][sampler_name].sample((num_samples,))
            except:
                warnings.warn(f"Could not sample from the sampler. Please define a valid sampler to sample from using the '.set_sampler_for_subspace()' method", stacklevel=2)
        
        # Ensure data structure compatibility with the SE
        samples = torch.as_tensor(samples).float()
        
        subspace_estimator = model["subspace_estimator"]
        
        # Warn if an untrained network is being used
        if not subspace_estimator.net_has_trained[feature_index]:
            warnings.warn(f"Network {feature_index} of '{name}' has not been fully trained. The basis vectors obtained is not informed [likely random from the network weights initialization]", stacklevel=2)
        
        # Get and return the active subspace basis
        active_subspace_basis = subspace_estimator.get_basis_vector(samples,feature_index, num_eigen_vect = num_eigen_vect_per_feature)
        if not return_samples:
            return active_subspace_basis
        else:
            return [active_subspace_basis, samples]


    def get_DRFFIT_subspace_from_subspace_estimator(self, samples = None, num_samples = 10000, sample_from_default_sampler_state = True, name = "default", sampler_name = 'default',num_eigen_vect_per_feature = 1, return_samples = False, *args, **kwargs):
        """     This function obtains the basis vector(s) of the DRFFIT subspace using the regression network of the subspace estimator (SE)
        Args:
            samples:                                Parameter samples used for estimation of the basis vector(s) of the DRFFIT subspace (if none, they are taken from the sampler)
            num_samples:                            Specify the number of samples to take from the sampler if not samples are passed
            sample_from_default_sampler_state:      Define if the samples are to be taken from default state of the sampler
            name:                                   The name of the SE object
            sampler_name:                           The name of the sampler to be used it no samples are passed
            num_eigen_vect_per_feature:             Specify the number of eigenvectors of the active subspace for each feature to use in the DRFFIT subspace 
            return_samples:                         Define if the samples are to be returned along with the subspace
            **kwargs:                               Only to collect additional arguments (avoid crashing)
        """

        # Get the SE
        model = self.get_subspace_estimator(name)
        if model == None:
            return

        if name == 'default':
            name = self.default_subspace_estimator_name

        if samples == None:
            if not sample_from_default_sampler_state:
                try:
                    self.sampler[name][sampler_name].set_state(**kwargs)
                except:
                    try:
                        self.sampler[name][sampler_name].set_default_x(**kwargs)
                    except:
                        warnings.warn(f"Could not set the sampler state using 'set_state' nor 'set_default_x', using default sampler state.", stacklevel=2)
            try:
                samples = self.sampler[name][sampler_name].sample((num_samples,))
            except:
                warnings.warn(f"Could not sample from the sampler. Please define a valid sampler to sample from using the '.set_sampler_for_subspace()' method", stacklevel=2)
            
        # Ensure data structure compatibility with the SE
        samples = torch.as_tensor(samples).float()
        
        subspace_estimator = model["subspace_estimator"]
        
        # Warn if an untrained network is being used (and which)
        if not np.array(subspace_estimator.net_has_trained).all():
            ind = np.arange(len(subspace_estimator.net_has_trained))[np.logical_not(np.array(subspace_estimator.net_has_trained))]
            warnings.warn(f"'{name}' has not been fully trained. The basis vectors obtained from subspace_estimator '{ind}' are not informed [likely random from the subspace_estimator weights initialization]", stacklevel=2)
        # Get and return the DRFFIT subspace
        DRFFIT_subspace = subspace_estimator.get_subspace(samples,num_eigen_vect_per_feature = num_eigen_vect_per_feature, **kwargs)
        samples = samples.detach()
        if not return_samples:
            return DRFFIT_subspace
        else:
            return [DRFFIT_subspace, samples]

    # TODO: Handle warnings to avoid it crashing and inform correct usage
    def sample_in_subspace(self, center_point, sample_width, sample_size, sample_distribution = 'sphere', target = None, subspace_estimator_name = 'default', *args, **kwargs):
        """     This function samples uniformly from a DRFFIT subspace 
        Args:
            center_point:                           Point around which the samples will be taken
            sample_width:                           Width of the sample space (Scaling of the basis)
            sample_size:                            Number of samples to return
            sample_distribution:                    Defines if the samples are taken from hypercube or hypersphere
            subspace_estimator_name:                The name of the SE object
            **kwargs:                               Collecting arguments that will be passed to the '.get_DRFFIT_subspace_from_subspace_estimator()' function
        """
        target_feature_values = None
        # Get DRFFIT subspace
        subspace = self.get_DRFFIT_subspace_from_subspace_estimator(name = subspace_estimator_name,sample_from_default_sampler_state = False, width = sample_width, point = center_point, **kwargs).detach()
        try:
            center_point = center_point.detach().numpy()
        except:
            pass
        if sample_distribution == 'sphere':
            # Sample uniformly in the -0.5 to 0.5 range for all subspace dimensions
            samples_surface = np.random.uniform(size = (sample_size, subspace.shape[0])) - 0.5
            # Normalize length to get unit vectors
            samples_surface /= np.linalg.norm(samples_surface, axis = 1).reshape((-1,1))
            # Generate randomized lengths for the unit vectors (0-1 range)
            samples_length = np.sqrt(np.random.uniform(size = (sample_size, 1)))
            # Generate the samples vectors by multiplying the unit vectors with their length
            samples_vectors = torch.as_tensor(np.tile((samples_surface*samples_length).reshape(sample_size, subspace.shape[0],1), self.theta_dim))
        else:
            # Generate samples uniformly in the -0.5 to 0.5 range as subspace coordinates
            samples_vectors = torch.as_tensor(np.tile(np.random.uniform(size = (sample_size, subspace.shape[0],1)) - 0.5, self.theta_dim))

        # Linear combinations of the sample vectors to return the samples in the original parameter space dimensionality (0-1 normalized space)
        search_vectors = (samples_vectors * subspace).detach().numpy().sum(axis = 1) 

        # Rescale samples
        search_vectors *= sample_width #* 0.01
        

        # Define the bounds (Used to perform the operations in a normalized space and to limit the samples to the bounded search space)
        lower_bound = self.theta_min.detach().numpy()
        upper_bound = (self.theta_min + self.theta_range).detach().numpy()
        par_range = upper_bound-lower_bound

        # Bring the point into the 0-1 normalized space
        point = (center_point - lower_bound) / par_range

        # Add the vectors to the point
        samples_scaled = search_vectors+point

        # Transfer the samples to the original parameter space 
        samples = samples_scaled * par_range + lower_bound

        # Reject limit the values of the parameter values to contain the samples in the bounded search space 
        samples = np.where(samples > upper_bound, upper_bound, samples)
        samples = np.where(samples < lower_bound, lower_bound, samples)

        return ensure_torch(samples)
    
    
    # TODO: Allow switching the sampler from default to non default as possible with the data, subspace_estimators, etc.
    def set_sampler_for_subspace(self, sampler, sampler_name = 'default', subspace_estimator_name = 'default', enforce_replace = None,*args,**kwargs):
        """     This function stores and sets a sampler for subspace estimation within the DRFFIT object (e.g., a uniform around sampler)
        Args:
            sampler:                   The sampler object
            sampler_name:              The name to be given to the sampler object
            subspace_estimator_name:   The name of the subspace estimator for which the sampler is to be attached to
            enforce_replace:           Enforce a reset on the network parameters if the network has already been trained
            **kwargs:                  To collect additional arguments (avoid crashing) 
        """
        
        model = self.get_subspace_estimator(subspace_estimator_name)
        if model == None:
            return
        if subspace_estimator_name == 'default':
            subspace_estimator_name = self.default_subspace_estimator_name
        try:
            test_samples = sampler.sample((100,))
        except:
            warnings.warn(f"Sampler does not have a '.sample()' method or raised an error when calling '.sample()'. Please provide a sampler that can be sampled from using a '.sample()' method", stacklevel=2)
            return
        if test_samples.shape[1] != self.theta_dim:
            warnings.warn(f"Sampler does not return parameters of the right dimension. Please provide a sampler that returns valid parameters.", stacklevel=2)
            return
        try:
            _ = self.sampler[subspace_estimator_name][sampler_name]
            if enforce_replace == None:
                warnings.warn(f"There is already a sampler for '{subspace_estimator_name}'. Please specify if it needs to be replaced by the new one using the 'enforce_replace' argument.", stacklevel=2)
            if enforce_replace:
                self.sampler[subspace_estimator_name][sampler_name] = sampler
            else:
                return
        except:
            self.sampler[subspace_estimator_name][sampler_name] = sampler


    def uniform_add_feature_fn_to_subspace_estimator(self, feature_fn_name = "default_AE", subspace_estimator_name = "default", enforce_replace = False, *args, **kwargs):
        """     This function attaches a feature function to a subspace estimator (SE) giving all features the same regression network architecture
        Args:
            name:                       Name of the feature function
            subspace_estimator_name:    Name of the SE
            enforce_replace:            Define if the network is to be replaced if it already exists
            **kwargs:                   Collect arguments to pass into .add_features() of the SE object such as a specific architecture
        """

        # Get the SE
        model = self.get_subspace_estimator(subspace_estimator_name)
        if model == None:
            return
        
        # Get the fn
        fn_all = self.get_fn(feature_fn_name, get_feature_list=True)
        if fn_all == None:
            return
        feature_list = fn_all[1]
        
        # Add all the features from the function to the SE
        try:
            for n, _ in model["features"]:
                if n == feature_fn_name:
                    if not enforce_replace:
                        warnings.warn(f"'{subspace_estimator_name}' subspace_estimator already has the features from '{feature_fn_name}'", stacklevel=2)
                        return
                    else:
                        model["subspace_estimator"].add_features(len(feature_list), enforce_replace_all = enforce_replace, **kwargs)
                        return
            for feature_index in feature_list:
                model["features"].append((feature_fn_name, feature_index))
        except:
            warnings.warn(f"'{subspace_estimator_name}' is an invalid subspace_estimator [it has no list of features]", stacklevel=2)
            return
        #model["subspace_estimator"].add_features(len(feature_list), **kwargs)
        try:
            model["subspace_estimator"].add_features(len(feature_list), **kwargs)
            return
        except:
            warnings.warn(f"'{subspace_estimator_name}' is an invalid subspace_estimator [it has no list of features]", stacklevel=2)
            return


    def add_custom_fn(self, fn, num_features, name, replace = False, *args, **kwargs):
        """     This function adds a user custom feature function to the DRFFIT object
        Args:
            fn:                         An object wrapper for the feature function. Must have a function called .feature_fn(x) that returns the feature values 
            num_features:               The number of features returned by .feature_fn(x)
            name:                       Name of the feature function
            replace:                    Specify to replace if a function of that name already exists in the DRFFIT object
            **kwargs:                   Collect additional arguments (avoid crashing)
        """

        if not replace:
            try:
                _ = self.feature_fn[name]
                warnings.warn(f"'{name}' already exists. Select replace = True if you want to replace the existing one", stacklevel=2)
                return
            except:
                pass
        
        self.feature_fn_name.append(name)
        self.feature_fn[name] = {"fn":fn, "features":[i for i in range(num_features)]}
    

    def add_AE(self, num_features = 8, num_units = 128, name = "default_AE", pre_trained = False, model_path = "./", file_name = 'default_AE', replace = False, device = 'cpu', *args, **kwargs):
        """     This function creates and adds an AutoEncoder to the DRFFIT object
        Args:
            num_features:               The number of features returned by .feature_fn(x)
            num_units:                  Specify the number of units in the first layer given the default architecture
            name:                       Name of the AE 
            pre_trained:                Specify to use a pretrained AE
            model_path:                 Specify the path where the state dict is stored
            replace:                    Specify to overwrite the current AE under the 'name' if the it already exists
            **kwargs:                   Will be passed in the AutoEncoder constructor. Can be used to specify an architecture  
        """

        # Handle replace
        if not replace:
            try:
                _ = self.feature_fn[name]
                warnings.warn(f"'{name}' already exists. Select replace = True if you want to replace the existing one", stacklevel=2)
                return
            except:
                pass
        # Create the AutoEncoder
        self.feature_fn_name.append(name)
        self.feature_fn[name] = {"fn":AutoEncoder(input_size = self.output_dim, num_features = num_features, num_units = num_units, device = device, **kwargs).to(device), "features":[i for i in range(num_features)]}
        
        # Load the weights
        if pre_trained:
            try:
                self.feature_fn[name]["fn"].load(model_path, file_name, device = device)
                self.feature_fn[name]["features"] = [i for i in range(self.feature_fn[name]["fn"].num_features)]
                self.feature_fn[name]["fn"] = self.feature_fn[name]["fn"].to(device)
                return
            except:
                try: # Verify that the file exists
                    f = open(model_path+file_name+".pkl", "r")
                    f.close()
                    warnings.warn("Incompatible model architecture, you can attempt to load the model again through the 'load_AE()' method", stacklevel=2)
                except:
                    warnings.warn("Invalid model path, you can attempt to load the model again through the 'load_AE()' method", stacklevel=2)
                return


    def train_feature_fn(self, name = 'default_AE', verbose = -1, enforce_replace = None, data = 'default', data_name = 'default',return_val = False, *args, **kwargs):
        """     This function trains a stored feature within the DRFFIT object (e.g., an AutoEncoder)
        Args:
            name:                      The name of the feature function object
            verbose:                   Allow a specified verbose state to overwrite the default
            enforce_replace:           Enforce a reset on the network parameters if the network has already been trained
            data:                      The name of the group of datasets
            data_name:                 The name of the dataset 
            **kwargs:                  Will be passed to the .train() function  
        """

        model = self.get_fn(name)
        if model == None:
            return
        if enforce_replace == None and model.has_trained:
            warnings.warn(f"Failed to train '{name}' feature function: it has already been trained.\n Pleace specify if you want to replace the current models or train from the current state using the 'enforce_replace' argument. Note that if it is an Autoencoder, you can save the current version before training or replacing by calling the 'save_AE()' function", stacklevel=2)
            return
        if enforce_replace == True:
            model.reset()

        try: # If the group of datasets sexists
            _ = self.data[data]
        except:
            warnings.warn(f"'{data}' is not a known set of datasets. You can add a set of data with the 'add_data()' method ", stacklevel=2)
            return
        
        try: # If the dataset exists
            dataset = self.data[data][data_name]
        except:
            warnings.warn(f"{data_name} is not in '{data}'", stacklevel=2)
            return

        # Set the data for training
        try:
            model.set_data(dataset["x"], **kwargs)
        except:
            try:
                model_type = model.get_type()
                if model_type == "AE" or model_type == "PCA":
                    warnings.warn("There is an issue with the model set_data method", stacklevel=2)
                    return
            except:
                warnings.warn(f"There is an issue with the '.set_data()' method of '{name}'. And it is not a built-in feature function type", stacklevel=2)
                return
        
        # Set default verbose state
        if verbose == -1:
            verbose = self.default_verbose
        
        # Train the model
        try:
            if return_val:
                val = model.train(verbose = verbose,return_val = return_val, **kwargs)
                return val
            model.train(verbose = verbose, **kwargs)
        except:
            try:
                model_type = model.get_type()
                if model_type == "AE" or model_type == "PCA":
                    warnings.warn(f"There is an issue with the '.train()' method of '{name}'", stacklevel=2)
                    return
            except:
                warnings.warn(f"There is an issue with the '.train()' method of '{name}'. And it is not a built-in feature function type", stacklevel=2)
                return


    def add_data(self, x, theta = None, features = None, name = "default", data_name = "default", *args, **kwargs):
        """     This function adds a dataset to the DRFFIT object
        Args:
            x:               Simulation output data or empirical observations data
            theta:           Parameter values if x is simulated
            features:        Feature values for the dataset if any were computed already
            name:            Name of the set of datasets 
            data_name:       Name of the dataset
            **kwargs:        Collect additional arguments (avoid crashing)  
        """

        x = torch.as_tensor(x).float()
        try:
            self.data[name][data_name] = {"x":x,"theta":theta,"features":features}
        except:
            self.data[name] = {data_name:{"x":x,"theta":theta,"features":features}}


    def del_data(self, name, data_name = None, *args, **kwargs):
        """     This function deletes a dataset or a set of datasets from the DRFFIT object
        Args:
            name:            Name of the set of datasets that contains the dataset
            data_name:       Name of the dataset (the whole set is deleted if it remains = None)
            **kwargs:        Collect additional arguments (avoid crashing)   
        """

        if data_name == None:
            try:
                self.data.pop(name)
                return
            except:
                return
        try:
            self.data[name].pop(data_name)
            return
        except:
            return


    def get_fn(self, name, get_feature_list = False):
        """     This internal function returns a feature function
        Args:
            name:               Name of the feature function
            get_feature_list:   Defines if the list of features computed by the feature function is to be returned
        """

        try:
            feature_fn = self.feature_fn[name]
        except:
            warnings.warn(f"'{name}' is not a known feature function", stacklevel=2)
            return None
        try:
            model = feature_fn["fn"]
        except:
            warnings.warn(f"'{name}' has no function", stacklevel=2)
            return None
        try:
            features = feature_fn["features"]
            if get_feature_list:
                return [model, features]
            else:
                return model
        except:
            warnings.warn(f"'{name}' has no feature(s)", stacklevel=2)
            return None      
    

    def get_subspace_estimator(self, name):
        """     This internal function returns a subspace estimator
        Args:
            name:               Name of the subspace estimator
        """

        try:
            subspace_estimator = self.subspace_estimator[name]
            return subspace_estimator
        except:
            warnings.warn(f"'{name}' is not a known subspace_estimator", stacklevel=2)
            return None 


    def save_AE(self, model_path, file_name, name = "default_AE", enforce_replace = False):
        """     This function saves an AutoEncoder's state dict
        Args:
            model_path:      The path where to save the AE's state dict (contains the name of the file too)
            name:            Name of the AE to save
        """

        model = self.get_fn(name)
        if model == None:
            return 
        try:
            model.save(model_path, file_name, enforce_replace = enforce_replace)
        except:
            try:
                model_type = model.get_type()
                if model_type == "AE":
                    warnings.warn("Invalid model path", stacklevel=2)
                    return
                else:
                    warnings.warn(f"'{name}' is not an AutoEncoder", stacklevel=2)
                    return
            except:
                warnings.warn(f"'{name}' is not an AutoEncoder and does not have a get_type() method", stacklevel=2)
                return

    
    def load_AE(self, model_path, file_name, name = "default_AE"):
        """     This function saves an AutoEncoder's state dict
        Args:
            model_path:      The path that contains the pretrained AE's state dict (contains the name of the file too)
            name:            Name of the AE that will load the pretrained weights
        """

        model = self.get_fn(name)
        if model == None:
            return
        try:
            model.load(model_path, file_name)
        except:
            try:
                model_type = model.get_type()
                if model_type == "AE":
                    try:
                        f = open(model_path+file_name+".pkl", "r")
                        f.close()
                        warnings.warn("Incompatible model architecture, you can attempt to load the model again through the 'load_AE()' method", stacklevel=2)
                    except:
                        warnings.warn("Invalid model path, you can attempt to load the model again through the 'load_AE()' method", stacklevel=2)
                else:
                    warnings.warn(f"'{name}' is not an AutoEncoder", stacklevel=2)
                    return
            except:
                warnings.warn(f"'{name}' is not an AutoEncoder and does not have a get_type() method", stacklevel=2)
                return

    
    def save_subspace_estimator(self, model_path, name = "default", save_data = False):
        """     This function saves a subspace estimator's regression network's state dicts
        Args:
            model_path:      The path where to save the state dicts (contains the name of the file too)
            name:            Name of the SE to save
            save_data:       Defines if the data contained in the DRFFIT object is to be saved along with the subspace estimator (all the data will be saved, use 'del_data()' _                                                                                                                     to remove the unwanted data before saving)
        """

        if model_path[-4:] != '.npy':
            warnings.warn(f"Failed to save '{name}' subspace_estimator: The model is a list of neural networks whose state_dict() are saved within a numpy array of objects, please enter a path name ends with '.npy'.", stacklevel=2)
            return
        model = self.get_subspace_estimator(name)
        if model == None:
            return 
        try:
            state_dicts = model['subspace_estimator'].get_all_state_dicts()
        except:
            try:
                model_type = model['subspace_estimator'].get_type()
                if model_type == "subspace_estimator":
                    warnings.warn("Unable to get the state dicts of the subspace_estimator's networks", stacklevel=2)
                    return
                elif model_type == "AE":
                    warnings.warn("This is an AutoEncoder, please use 'save_AE' to save this model", stacklevel=2)
                    return
                else:
                    warnings.warn(f"'{name}' is not a 'subspace_estimator'", stacklevel=2)
                    return
            except:
                warnings.warn(f"'{name}' is not a 'subspace_estimator' and does not have a get_type() method", stacklevel=2)
                return
        if len(state_dicts) != len(model['features']):
            warnings.warn(f"Failed to save '{name}' subspace_estimator: There is a mismatch between the number of state_dict() returned by the '{name}' subspace_estimator object and the number of features listed in the model's info", stacklevel=2)
        data = None
        if save_data:
            data = self.data
        save_array = []
        for i in range(len(state_dicts)):
            save_array.append({"state_dict":state_dicts[i], "feature": model['features'][i]})
        save_array = np.array([save_array, data], dtype = object)
        try:
            np.save(model_path, save_array, allow_pickle = True)
        except:
            warnings.warn(f"Failed to save '{name}' subspace_estimator: Invalid path", stacklevel=2)


    def save_SE_as_log(self, model_path = "./", file_name = 'default_AE_SE', name = "default", enforce_replace = False):
        """     This function saves the weights of all the regression networks of a subspace estimator
        Args:
            model_path:             The path where to load the SE 
            file_name:              The name of the file that contains the saved SE
            name:                   Name of the SE that will load the state SE
            enforce_replace:        Defines if the SE's weights need to be replaced if it already exist
        """

        model = self.get_subspace_estimator(name)
        if model == None:
            return

        try:
            f = open(model_path+'/'+file_name+'.pkl', "rb")
            f.close()
            if not enforce_replace:
                warnings.warn("File already exists, please select 'enforce_replace = True' to overwrite it.", stacklevel=2)
                return
        except:
            try:
                model['subspace_estimator'].save_as_log(model_path, file_name, enforce_replace = enforce_replace)
            except:
                warnings.warn("Error in '.save_as_log()'", stacklevel=2)
                return
        
    def load_SE_from_log(self, model_path = "./", file_name = 'default_AE_SE', name = "default", enforce_replace = False, device = 'cpu'):
        """     This function loads the weights of all the regression networks of a subspace estimator
        Args:
            model_path:             The path where to load the SE 
            file_name:              The name of the file that contains the saved SE
            name:                   Name of the SE that will load the state SE
            enforce_replace:        Defines if the SE's weights need to be replaced if it already exist
        """

        model = self.get_subspace_estimator(name)
        if model == None:
            return
        if not enforce_replace and len(model['subspace_estimator'].net_has_trained)> 0 and any(model['subspace_estimator'].net_has_trained):
            warnings.warn(f"Failed to load '{file_name}' subspace_estimator: The subspace_estimator contains features that have already been trained\n Pleace specify 'enforce_replace = True' to replace '{name}'", stacklevel=2)
            return
        try:
            model['subspace_estimator'].load_as_log(model_path, file_name, device = device)
        except:
            warnings.warn("Invalid model path or error in .load_as_log(), you can attempt to load the model again through the 'load_subspace_estimator()' method", stacklevel=2)
            return
            
    def load_subspace_estimator(self, model_path, name = "default", enforce_replace = None, load_data = False, enforce_replace_data = None):
        """     This function loads the weights of all the regression networks of a subspace estimator
        Args:
            model_path:             The path where to save the state dicts (contains the name of the file too)
            name:                   Name of the SE that will load the pretrained weights
            enforce_replace:        Defines if the SE's weights need to be replaced if it already exist
            load_data:              Defines if the data contained in the saved file is to be loaded along with the subspace estimator
            enforce_replace_data:   Defines if the current data is to be replaced by the loaded data if it already exists

        """

        if model_path[-4:] != '.npy':
            warnings.warn(f"Failed to save '{name}' subspace_estimator: The model is a list of neural networks whose state_dict() are saved within a numpy array of objects, please enter a path name ends with '.npy'.", stacklevel=2)
            return
        model = self.get_subspace_estimator(name)
        if model == None:
            return
        try:
            save_array = np.load(model_path, allow_pickle = True)
        except:
            warnings.warn("Invalid model path, you can attempt to load the model again through the 'load_subspace_estimator()' method (note that the model is saved as a numpy array, make sure the path name ends with '.npy')", stacklevel=2)
            return

        already_trained = []
        for i in range(len(model['subspace_estimator'].net_has_trained)):
            if model['subspace_estimator'].net_has_trained[i]:
                already_trained.append(model['features'][i])
        if enforce_replace == None:
            if len(already_trained)>0:
                warnings.warn(f"Failed to load '{name}' subspace_estimator: The subspace_estimator networks for the following features have already been trained: {already_trained} \n Pleace specify if you want to replace the current models or not using the 'enforce_replace' argument", stacklevel=2)
                return
            enforce_replace = False
        saved_dict = save_array[0]
        saved_data = save_array[1]
        if load_data and saved_data != None:
            for data in saved_data:
                try:
                    _ = self.data[data]
                    if enforce_replace_data == None:
                        warnings.warn(f"Failed to load '{data}' data from saved subspace_estimator:\n Pleace specify if you want to replace the current data with that name or not using the 'enforce_replace_data' argument", stacklevel=2)
                        return
                    if enforce_replace_data:
                        self.data[data] = saved_data[data]
                except:
                    self.data[data] = saved_data[data]
        model_dicts = []
        subspace_estimator_features = []
        for i in range(len(saved_dict)):
            model_dicts.append(saved_dict[i]['state_dict'])
            subspace_estimator_features.append(saved_dict[i]['feature'])
        try:
            model['subspace_estimator'].load_all_from_state_dicts(model_dicts, replace = enforce_replace)
            model['features'] = subspace_estimator_features
        except:
            try:
                model_type = model['subspace_estimator'].get_type()
                if model_type == "subspace_estimator":
                    try:
                        f = open(model_path, "r")
                        f.close()
                        warnings.warn("Incompatible model architecture in at least one of the networks, you can attempt to load the model again through the 'load_subspace_estimator()' method", stacklevel=2)
                    except:
                        warnings.warn("Invalid model path, you can attempt to load the model again through the 'load_subspace_estimator()' method (note that the model is saved as a numpy array, make sure the path name ends with '.npy')", stacklevel=2)
                elif model_type == "AE":
                    warnings.warn("This is an AutoEncoder, please use 'save_AE' to save this model", stacklevel=2)
                    return
                else:
                    warnings.warn(f"'{name}' is not a 'subspace_estimator'", stacklevel=2)
                    return
            except:
                warnings.warn(f"'{name}' is not a 'subspace_estimator' and does not have a get_type() method", stacklevel=2)
                return


    def subspace_estimator_info(self, name = "default"):
        """     This function displays the summary of a SE 
        Args:
            name:   Name of the SE
        """

        output = {"name":name}
        if name == "default" or name == self.default_subspace_estimator_name:
            name = "default"
            output["name"] = f"default ({self.default_subspace_estimator_name})"
        model = self.get_subspace_estimator(name)
        if model == None:
            return
        num_features = len(model["features"])
        regression_networks = model["subspace_estimator"].net_list
        has_trained = model["subspace_estimator"].net_has_trained
        output["device"] = model["subspace_estimator"].device
        output["data"] = {"Stored data": [data for data in model["subspace_estimator"].prev_data], "Current": model["subspace_estimator"].curr_data}

        networks_info = {}
        for i,rn in enumerate(regression_networks):
            networks_info[model["features"][i]] = {
                "Input size": regression_networks[i].input_size,
                "Architecture": regression_networks[i].architecture,
                #"Ceil output":regression_networks[i].ceil_output,
                "Trained": has_trained[i]

                }
        for info in output:
            print(info)
            print(f"\t{output[info]}")
        print("Networks info")
        for info in networks_info:
            print(f"\tFeature: {info}")
            for details in networks_info[info]:
                print(f"\t\t{details}: {networks_info[info][details]}")


    def _info(self, subspace_estimator_details = True):
        """     This function displays the summary of the DRFFIT object 
        Args:
            subspace_estimator_details:   Defines if the details of all the existing SE are to be displayed
        """

        all_estimators = [subspace_estimator for subspace_estimator in self.subspace_estimator]
        separator_width = 100
        print(f"Data:")
        for d in self.data:
            print(f"\tData '{d}', which contains:")
            for set in self.data[d]:
                print(f"\t\tDataset '{set}' with {self.data[d][set]['x'].shape[0]} instances: ", end = "")
                for data_type in self.data[d][set]:
                    try:
                        data_shape = self.data[d][set][data_type].shape
                        print(f"{data_type} of shape {data_shape[1:]}, ", end = "")
                    except:
                        print(f"no {data_type}, ", end = "")
                        pass
                print("\n", end = "")
        
        for fn in self.feature_fn:
            print(f"\n{'='*separator_width}")
            print(f"Feature function '{fn}', which contains features {self.feature_fn[fn]['features']}")
            print(f"\n{self.feature_fn[fn]['fn']}")
            print(f"{'='*separator_width}")

        print(f"Available subspace estimators: \n\t{all_estimators}")
        if subspace_estimator_details:
            print("Subspace estimator details:")
            for subspace_estimator in self.subspace_estimator:
                print(f"\n{'='*separator_width}")
                self.subspace_estimator_info(subspace_estimator)
                print(f"{'='*separator_width}")
                

    

