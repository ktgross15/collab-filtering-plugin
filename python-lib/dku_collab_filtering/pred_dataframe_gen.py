import pandas as pd
from surprise import accuracy
from dataiku.customrecipe import *

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

def create_ratings_df(preds, user_id_col, item_id_col):
    ratings_df_data = []
    for row in preds:
        # print row
        row_dict = {}
        row_dict[item_id_col] = row.iid
        row_dict[user_id_col] = row.uid
        row_dict['actual_rating'] = row.r_ui
        row_dict['pred_rating'] = row.est
        ratings_df_data.append(row_dict)
    preds_df = pd.DataFrame(ratings_df_data, columns=[user_id_col, item_id_col, 'actual_rating', 'pred_rating'])
    return preds_df

def get_pivoted_df(master_df, user_id_col, item_id_col, value):
    pivoted_df = master_df.pivot(index=user_id_col, columns=item_id_col, values=value)
    return pivoted_df

def merge_preds_actual_dfs(pred_df, actual_df):
    pred_df.columns = [str(col) + '_pred' for col in pred_df.columns]
    actual_df.columns = [str(col) + '_actual' for col in actual_df.columns]
    full_df = pd.merge(pred_df, actual_df, left_index=True, right_index=True)
    return full_df


# def generate_train_test_preds(model, train, test):
#     test_preds = model.test(test)
#     train_set_ready = train.build_testset()
#     train_preds = model.test(train_set_ready)
#     return test_preds, train_preds

def generate_test_score(test_preds, error_metric):
    if error_metric == 'rmse':
        return accuracy.rmse(test_preds)
    elif error_metric == 'mae':
        return accuracy.mae(test_preds)
    elif error_metric == 'fcp':
        return accuracy.fcp(test_preds)
    # else??