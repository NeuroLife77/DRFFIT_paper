from sklearn.decomposition import PCA
import numpy as np
from torch import as_tensor
from lib.utils import ensure_numpy, ensure_torch

class PCA_features:
    
    def __init__(self, num_features, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, apply_basis_change_on_features = False, *args, **kwargs):
        self.num_features = num_features
        self.args = {"copy":copy,"whiten":whiten,"svd_solver":svd_solver, "tol":tol, "iterated_power":iterated_power, "random_state":random_state}
        self.pca_model = PCA(self.num_features,**self.args)
        self.has_trained = False
        self.has_data = False
        self.data = None
        self.change_of_basis = np.eye(self.num_features)
        self.apply_basis_change_on_features = apply_basis_change_on_features
        if self.apply_basis_change_on_features:
            self.change_of_basis += np.random.randn(self.change_of_basis.shape[0],self.change_of_basis.shape[1])
            self.change_of_basis /= np.linalg.det(self.change_of_basis)
    
    
    def set_data(self,x, *args, **kwargs):
        try:
            data = x.detach().numpy()
        except:
            data = np.array(x)
        self.data = data
        self.has_data = True

    def train(self, *args, **kwargs):
        self.pca_model.fit(self.data)
        self.has_trained = True

    def feature_fn(self, x, *args, **kwargs):
        x = ensure_numpy(x)
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        features = self.pca_model.transform(x) @ self.change_of_basis
        features = ensure_torch(features)
        return features
    
    def reset(self, *args, **kwargs):
        self.pca_model = PCA(self.num_features,**self.args)
        self.change_of_basis = np.eye(self.num_features)
        if self.apply_basis_change_on_features:
            self.change_of_basis += np.random.rand(self.change_of_basis.shape[0],self.change_of_basis.shape[1])
            self.change_of_basis /= np.linalg.det(self.change_of_basis)

    def get_type(self, *args, **kwargs):
        return "PCA"