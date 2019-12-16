from dataiku.customrecipe import *

def read_recipe_config(param):
    param_val = get_recipe_config().get(param, '')
    return param_val
