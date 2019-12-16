from dataiku.customrecipe import *
import pickle

def read_recipe_config(param):
    param_val = get_recipe_config().get(param, '')
    return param_val


def dump_to_folder(folder, algo_name, predictions=None, algo=None):
    """A basic wrapper around Pickle to serialize a list of prediction and/or an algorithm on drive."""
    
    file_name = '{}_best_model.pkl'.format(algo_name)
    # maybe delete predicitons?
    dump_obj = {'predictions': predictions, 'algo': algo}
    with folder.get_writer(file_name) as file_obj:
        pickle.dump(dump_obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)