import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import datetime
from surprise import SVD, SVDpp, NMF, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, NormalPredictor, BaselineOnly, SlopeOne, CoClustering
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
from surprise import dump
import pickle 


def create_ratings_df(preds, u_id_col, i_id_col):
    iids = []
    uids = []
    act_rrs = []
    est_rrs = []

    for row in preds:
        iids.append(row.iid)
        uids.append(row.uid)
        act_rrs.append(row.r_ui)
        est_rrs.append(row.est)

    preds_df = pd.DataFrame(columns=[u_id_col,i_id_col,'actual_rr','pred_rr'])
    preds_df[u_id_col] = uids
    preds_df[i_id_col] = iids
    preds_df['actual_rr'] = act_rrs
    preds_df['pred_rr'] = est_rrs

    return preds_df

def get_pivoted_pred_df(surprise_algo_preds, u_id_col, i_id_col):
    master_df = create_ratings_df(surprise_algo_preds, u_id_col, i_id_col)
    predictions_df_pivoted = master_df.pivot(index=u_id_col, columns=i_id_col, values='pred_rr')
    return predictions_df_pivoted


#### READ INPUT PARAMETERS ####
def read_recipe_config(param):
    param_val = get_recipe_config().get(param, None)
    return param_val

user_id_col = read_recipe_config('user_id_col')
item_id_col = read_recipe_config('item_id_col')
rating_col = read_recipe_config('rating_col')
ratings_scale_min = read_recipe_config('ratings_scale_min')
ratings_scale_max = read_recipe_config('ratings_scale_max')
algo = read_recipe_config("selected_algoritm")


#### READ & FORMAT RECIPE INPUTS
input_dataset = dataiku.Dataset(get_input_names_for_role('input_dataset')[0])
input_df = input_dataset.get_dataframe()
input_df = input_df[[user_id_col, item_id_col, rating_col]]

reader = Reader(rating_scale = (ratings_scale_min, ratings_scale_max))
data = Dataset.load_from_df(input_df, reader)


#### SCORE FINAL MODEL ####

def load_modified(folder, model_file_name):
    with folder.get_download_stream(model_file_name) as file_obj:
        dump_obj = pickle.load(file_obj)
        model = dump_obj['predictions'], dump_obj['algo']
    return model

model_file_name = algo + '_best_model.pkl'
gs_folder = dataiku.Folder(get_input_names_for_role('grid_search_models')[0])
folder_type = gs_folder.get_info()['type']
model = load_modified(gs_folder, model_file_name)[1]
    
# train test split
train_set = data.build_full_trainset() # this is the full known ratings dataset - in Suprise-friendly format
train_set_ready = train_set.build_testset()
unknown_set = train_set.build_anti_testset() # this is the complement of train_set - all unknown possible user/item rating combinations

# generate predictions
train_preds = model.test(train_set_ready)
unknown_set_preds = model.test(unknown_set)
all_preds = train_preds + unknown_set_preds

# generate predictions dataframe
all_preds_pivoted = get_pivoted_pred_df(all_preds, user_id_col, item_id_col)
all_preds_pivoted.reset_index(inplace=True)


# Write recipe outputs
all_preds_dataset = dataiku.Dataset(get_output_names_for_role("all_predictions")[0])
all_preds_dataset.write_with_schema(all_preds_pivoted)
