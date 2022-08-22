from lib.utils import get_log, save_log

data_path = '/meg/meg1/users/dboutet/DRFFIT_env/DRFFIT/Data/HH/targets/tests/full_range/test_0/'
file_name = 'trial_1'

def get_targets(path = data_path, file = file_name):
    if 'targets' not in file:
        file+='_targets'
    targets = get_log(path, file)
    return targets['targets']['x'], targets['targets']['theta']