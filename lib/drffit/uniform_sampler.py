from torch import as_tensor
from lib.utils import ensure_torch, ensure_numpy
import numpy as np
from time import time

class uniform_around_sampler:
    def __init__(self,theta_min, theta_range = None, theta_max = None, random_seed = None,sample_distribution = 'sphere', **kwargs):
        theta_min = ensure_numpy(theta_min)
        theta_range = ensure_numpy(theta_range)
        theta_max = ensure_numpy(theta_max)
        self.theta_min = theta_min
        self.theta_dim = theta_min.shape[0]
        if theta_range is None and theta_max is not None:
            self.theta_max = theta_max
            self.theta_range = self.theta_max - self.theta_min
        elif theta_max is None and theta_range is not None:
            self.theta_range = theta_range
            self.theta_max = self.theta_min + self.theta_range
        else:
            self.theta_range = self.theta_min * 10
            self.theta_max = self.theta_min + self.theta_range
        self.center = self.theta_min + self.theta_range/2
        self.width = self.theta_range
        self.sample_distribution = sample_distribution
        if random_seed is None:
            np.random.seed(int(time()))
        kwargs = None
    
    def set_sample_center(self, point, **kwargs):
        point = ensure_numpy(point)
        self.center = point
    
    def set_sample_width(self,width, **kwargs):
        self.width = self.theta_range * width
        
    def set_state(self,point = None, width = None, **kwargs):
        if point is not None:
            self.set_sample_center(point)
        if width is not None:
            self.set_sample_width(width)

    def sample(self, sample_shape, **kwargs):
        try: 
            num_samples = sample_shape[0]
        except:
            num_samples = sample_shape
        if self.sample_distribution == 'sphere':
            # Sample uniformly in the -0.5 to 0.5 range for all subspace dimensions
            samples_surface = np.random.uniform(size = (num_samples, self.theta_dim)) - 0.5
            # Normalize length to get unit vectors
            samples_surface /= np.linalg.norm(samples_surface, axis= 1).reshape((-1,1))
            # Generate randomized lengths for the unit vectors (0-1 range)
            samples_length = np.sqrt(np.random.uniform(size = (num_samples, 1)))
            # Generate the samples vectors by multiplying the unit vectors with their length
            samples_vectors = samples_surface*samples_length
        else:
            # Generate samples uniformly in the -0.5 to 0.5 range as subspace coordinates
            samples_vectors = np.random.uniform(size = (num_samples, self.theta_dim)) - 0.5
        
        # Rescale samples
        samples_vectors *= self.width
        samples = self.center + samples_vectors
        samples = np.where(samples > self.theta_max, self.theta_max, samples)
        samples = np.where(samples < self.theta_min, self.theta_min, samples)
        
        return ensure_torch(samples)