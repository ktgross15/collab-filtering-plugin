import json
from read_inputs import *


# def add_param_to_algo_dict(algo_dictionary, algo_name, param, param_dict={}):    
#     algo_param_plugin = '{}_{}'.format(algo_name, param)
#     print "Adding parameters to algo dict:", param
    
#     # add parameter dictionary for sim or bsl options
#     if param in ['sim_options', 'bsl_options']:
#         print "adding parameter dictionary"
#         algo_dictionary[algo_name]['grid_params'][param] = param_dict
    
#     # add all other parameters
#     else:
#         param_val = read_recipe_config(algo_param_plugin)
#         if type(param_val) == bool:
#             param_val = [param_val]
#         elif (len(param_val) >= 1) & (param_val != "[]"):
#             param_val = json.loads(param_val)
#         algo_dictionary[algo_name]['grid_params'][param] = param_val
    
#     return algo_dictionary



# def add_algo_to_algo_dict(algo_dictionary, algo_bool, algo_mod, algo_name, params=[], param_dicts=[]):
#     if algo_bool:
#         print "ADDING", algo_name
#         # add algo to algo_dictionary
#         algo_dictionary[algo_name] = {}
#         algo_dictionary[algo_name]['grid_params'] = {}
#         algo_dictionary[algo_name]['module'] = algo_mod
        
#         # add each parameter to the grid search parameters
#         for param in params:
#             print param
#             algo_dictionary = add_param_to_algo_dict(algo_dictionary, algo_name, param)
        
#         # add parameter dictionaries if necessary
#         if len(param_dicts) >= 1:
#             for param_dict in param_dicts:
#                 algo_dictionary = add_param_to_algo_dict(algo_dictionary, algo_name, param, param_dict)
                
#     return algo_dictionary

# def add_algo_to_algo_dict(algo_dictionary, algo_bool, algo_mod, algo_name, params=[], param_dicts=[]):
#     if algo_bool:
#         print "ADDING", algo_name
#         # add algo to algo_dictionary
#         algo_dictionary[algo_name] = {}
#         algo_dictionary[algo_name]['grid_params'] = {}
#         algo_dictionary[algo_name]['module'] = algo_mod
                
#     return algo_dictionary

def generate_dict_from_params(algo_name, inner_params):
    param_dict = {}
    for inner_param in inner_params:
        algo_param_plugin = '{}_{}'.format(algo_name, inner_param)
        param_val = get_recipe_config().get(algo_param_plugin, '')
        print "got param val", inner_param, param_val, type(param_val)
        
        # if valid parameter_value is entered, add to parameter dict
        if type(param_val) == bool:
            param_dict[inner_param] = [param_val]
        elif (len(param_val) >= 1) & (param_val != "[]"):
            param_dict[inner_param] = [param_val]
         
    return param_dict