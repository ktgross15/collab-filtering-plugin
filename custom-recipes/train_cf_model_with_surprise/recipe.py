import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import datetime
import ast
import random
import json
import pickle 

from surprise import SVD, SVDpp, NMF, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, NormalPredictor, BaselineOnly, SlopeOne, CoClustering
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
from surprise.model_selection import GridSearchCV

from dku_collab_filtering.algo_dictionary import *
from dku_collab_filtering.read_inputs import *
from dku_collab_filtering.pred_dataframe_gen import *


#### READ INPUT PARAMETERS ####

user_id_col = read_recipe_config('user_id_col')
item_id_col = read_recipe_config('item_id_col')
rating_col = read_recipe_config('rating_col')
ratings_scale_min = read_recipe_config('ratings_scale_min')
ratings_scale_max = read_recipe_config('ratings_scale_max')
k_fold_splits = int(read_recipe_config('k_fold_splits'))
error_metric = read_recipe_config('error_metric')

svdpp_bool = read_recipe_config("svdpp")
svd_bool = read_recipe_config("svd")
nmf_bool = read_recipe_config("nmf")
normalpredictor_bool = read_recipe_config("normalpredictor")
baselineonly_bool = read_recipe_config("baselineonly")
slopeone_bool = read_recipe_config("slopeone")
coclustering_bool = read_recipe_config("coclustering")
knnbasic_bool = read_recipe_config("knnbasic")
knnbaseline_bool = read_recipe_config("knnbaseline")
knnwithmeans_bool = read_recipe_config("knnwithmeans")
knnwithzscore_bool = read_recipe_config("knnwithzscore")


#### READ AND FORMAT RECIPE INPUTS ####

input_dataset = dataiku.Dataset(get_input_names_for_role('input_dataset')[0])
input_df = input_dataset.get_dataframe()
input_df = input_df[[user_id_col, item_id_col, rating_col]]

# read input data into Surprise
reader = Reader(rating_scale = (ratings_scale_min, ratings_scale_max))
data = Dataset.load_from_df(input_df, reader)


def dump_modified(folder, file_name, predictions=None, algo=None):
    """A basic wrapper around Pickle to serialize a list of prediction and/or an algorithm on drive."""

    dump_obj = {'predictions': predictions, 'algo': algo}
    with folder.get_writer(file_name) as file_obj:
        pickle.dump(dump_obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
      
    
def test_model_return_prediction_sets(model, train, test, error_metric):
    test_preds = model.test(test)
    
    if error_metric == 'rmse':
        test_score = accuracy.rmse(test_preds)
    elif error_metric == 'mae':
        test_score = accuracy.mae(test_preds)
    elif error_metric == 'fcp':
        test_score = accuracy.fcp(test_preds)
    
    train_set_ready = train.build_testset()
    train_preds = model.test(train_set_ready)

    return test_preds, test_score, train_preds


#### GENERATE ALGO DICTIONARY WITH PARAMETERS FOR EACH ALGO INCLUDED ####

algo_dictionary = {}

# generate parameter dictionaries where necessary
baselineonly_bsl_dict = generate_dict_from_params(algo_dictionary, baselineonly_bool, 'baselineonly', ['method'])
knnbaseline_bsl_dict = generate_dict_from_params(algo_dictionary, knnbaseline_bool, 'knnbaseline', ['method'])
knnbasic_sim_options_dict = generate_dict_from_params(algo_dictionary, knnbasic_bool, 'knnbasic', ['name', 'user_based'])
knnbaseline_sim_options_dict = generate_dict_from_params(algo_dictionary, knnbaseline_bool, 'knnbaseline', ['name', 'user_based'])
knnwithmeans_sim_options_dict = generate_dict_from_params(algo_dictionary, knnwithmeans_bool, 'knnwithmeans', ['name', 'user_based'])
knnwithzscore_sim_options_dict = generate_dict_from_params(algo_dictionary, knnwithzscore_bool, 'knnwithzscore', ['name', 'user_based'])

# generate algorithm dictionaries
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, svd_bool, SVD, 'svd', ['n_factors', 'n_epochs', 'lr_all', 'reg_all', 'biased'])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, svdpp_bool, SVDpp, 'svdpp', ['n_factors', 'n_epochs', 'lr_all', 'reg_all'])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, nmf_bool, NMF, 'nmf', ['n_factors', 'n_epochs','biased'])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, normalpredictor_bool, NormalPredictor, 'normalpredictor')
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, slopeone_bool, SlopeOne, 'slopeone')
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, coclustering_bool, CoClustering, 'coclustering', ['n_cltr_u', 'n_cltr_i', 'n_epochs'])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, baselineonly_bool, BaselineOnly, 'baselineonly', ['bsl_options'], [baselineonly_bsl_dict])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, knnbasic_bool, KNNBasic, 'knnbasic', ['k', 'min_k', 'sim_options'], [knnbasic_sim_options_dict])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, knnbaseline_bool, KNNBaseline, 'knnbaseline', ['k', 'min_k', 'sim_options', 'bsl_options'], [knnbaseline_sim_options_dict, knnbaseline_bsl_dict])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, knnwithmeans_bool, KNNWithMeans, 'knnwithmeans', ['k', 'min_k', 'sim_options'], [knnwithmeans_sim_options_dict])
algo_dictionary = add_algo_to_algo_dict(algo_dictionary, knnwithzscore_bool, KNNWithZScore, 'knnwithzscore', ['k', 'min_k', 'sim_options'], [knnwithzscore_sim_options_dict])



#### MODEL TRAINING & GRID SEARCH ####

# train test split
raw_ratings = data.raw_ratings
random.shuffle(raw_ratings)

threshold = int(.8 * len(raw_ratings)) # A = 80% of the data, B = 20% of the data
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings
test_set = data.construct_testset(B_raw_ratings)  # testset is now the set B
train_set = data.build_full_trainset()


# generate models folder
folder_name = get_output_names_for_role('grid_search_models')[0]
surprise_cv_models = dataiku.Folder(folder_name)

# instantiate empty trained_models dictionary and master_cv_df dataframe
trained_models = {}
master_cv_df = pd.DataFrame()

# for each algorithm in the user-input list, train a model and add prediction datasets/error metrics to a master trained-models dictionary
all_algos = algo_dictionary.keys()
for algo in all_algos:
    model_module = algo_dictionary[algo]['module']
    param_grid = algo_dictionary[algo]['grid_params']
    print(model_module)
    print(param_grid)
    
    gs = GridSearchCV(model_module, param_grid, measures=[error_metric], cv=k_fold_splits, n_jobs=2, joblib_verbose=15)
    gs.fit(data)
    
    for estimator in gs.best_estimator.items():
        print(estimator)
        
        # retrain on the whole set A
        model = estimator[1]
        model.fit(train_set)

        model_file_name = algo + '_best_model.pkl'
        dump_modified(surprise_cv_models, model_file_name, algo=model)   
        
        model_test_preds, model_test_score, model_train_preds = test_model_return_prediction_sets(model, train_set, test_set, error_metric)
        
        name = algo + estimator[0]
        trained_models[name] = {}
        trained_models[name]['model'] = model
        trained_models[name]['model_test_preds'] = model_test_preds
        trained_models[name]['model_test_score'] = model_test_score
        trained_models[name]['model_train_preds'] = model_train_preds
            
    gs_results_df = pd.DataFrame.from_dict(gs.cv_results)
    gs_results_df['algorithm'] = algo
    
    print('Done with CV for algorithm: ' + algo)
    master_cv_df = master_cv_df.append(gs_results_df)
       
# modify gs output
# sort by error metric col and add new cols
metric_rank_col = 'rank_test_{}'.format(error_metric)
metric_val_col = 'mean_test_{}'.format(error_metric)
overall_col = 'overall_{}_rank'.format(error_metric)
metric_rank_within_algo = '{}_rank_within_algo'.format(error_metric)

master_cv_df = master_cv_df.sort_values(by=metric_val_col)
master_cv_df['added_to_models_folder'] = np.where(master_cv_df[metric_rank_col]==1, True, False)
master_cv_df = master_cv_df.rename(columns= {metric_rank_col: metric_rank_within_algo})
master_cv_df[overall_col] = master_cv_df[metric_val_col].rank()

# reorder gs dataframe col order
time_cols = []
error_cols = []
param_cols = []
other_cols = []

cols = master_cv_df.columns
for col in cols:
    if "time" in col:
        time_cols.append(col)
    elif (("rmse" in col) | ("mae" in col) | ("fcp" in col)):
        error_cols.append(col)
    elif "param" in col:
        param_cols.append(col)
    elif "algorithm" not in col:
        other_cols.append(col)

new_cols =  ['algorithm'] + param_cols + error_cols + time_cols + other_cols
master_cv_df = master_cv_df[new_cols]

# select the best model and get preds and score
best_model = min(trained_models, key=lambda x: trained_models[x].get('model_test_score'))
train_preds = trained_models[best_model]['model_train_preds']
test_preds = trained_models[best_model]['model_test_preds']
test_score = trained_models[best_model]['model_test_score']


##### FORMAT PREDICTION DATASETS ####

# generate pivoted dataframes with preds and actuals for both train and test
test_preds_df, test_actuals_df = get_pivoted_pred_act_dfs(test_preds, user_id_col, item_id_col)
train_preds_df, train_actuals_df = get_pivoted_pred_act_dfs(train_preds, user_id_col, item_id_col)

# merge preds and actuals to get compairison dataframes 
test_preds_actuals = merge_preds_actual_dfs(test_preds_df, test_actuals_df).reset_index()
train_preds_actuals = merge_preds_actual_dfs(train_preds_df, train_actuals_df).reset_index()

    
#### WRITE RECIPE OUTPUTS ####
surprise_cv_grid_search_results = dataiku.Dataset(get_output_names_for_role("grid_search_error_metrics")[0])
surprise_cv_grid_search_results.write_with_schema(master_cv_df)

train_preds_actuals_dataset = dataiku.Dataset(get_output_names_for_role("train_preds_actuals")[0])
train_preds_actuals_dataset.write_with_schema(train_preds_actuals)

test_preds_actuals_dataset = dataiku.Dataset(get_output_names_for_role("test_preds_actuals")[0])
test_preds_actuals_dataset.write_with_schema(test_preds_actuals)
