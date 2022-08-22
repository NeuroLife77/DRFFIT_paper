from torch import as_tensor, mean, from_numpy, no_grad
from torch.nn import CosineSimilarity
from pickle import load as pkload
from pickle import dump as pkldump

def save_log(log, directory, file_name, enforce_replace = False):
    try:
        with open(f'{directory}/{file_name}.pkl', 'rb') as f:
            _ = pkload(f)
        if not enforce_replace:
            print(f"The file '{directory}/{file_name}.pkl' already exists, please select 'enforce_replace = True' to rewrite it.")
            return
    except:
        pass
    with open(f'{directory}{file_name}.pkl', 'wb') as f:
        pkldump(log, f)


def mse(target, data, dim = 1):
    x = as_tensor(data).float()
    x_target = as_tensor(target).float()
    return mean((x-x_target)**2, dim = dim)

def correlation_loss_fn(target,candidates):
    target = target.view(1,-1)
    cos = CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(target - target.mean(dim=1,keepdim=True), candidates - candidates.mean(dim=1,keepdim=True))
    return -pearson
       
def get_log(directory, file_name):
    if directory is None and file_name == 'drffit_object':
        return None
    try:
        with open(f'{directory}{file_name}.pkl', 'rb') as f:
            log = pkload(f)
        return log
    except:
        print(f"File '{directory}{file_name}.pkl' does not exist.")
        return None

def make_list(thing):
    if isinstance(thing, list):
        for i, item in enumerate(thing):
            thing[i] = make_list(item)
    else:
        try:
            thing = thing.tolist()
        except:
            try:
                thing = thing.to_list()
            except:
                try:
                    thing = thing.data.item()
                except:
                    pass
    return thing

def ensure_torch(x, type_float = True):
    try:
        x = as_tensor(x)
        if type_float:
            x = x.float()
    except:
        try:
            x = from_numpy(x)
        except: 
            pass
    if type_float:
        try:
            x = x.float()
        except:
            pass
    return x

def ensure_numpy(x):
    
    try:
        x = x.detach()
    except:
        pass
    
    try:
        x = x.to('cpu')
    except:
        pass
    
    try:
        x = x.numpy()
    except:
        pass
    
    return x
