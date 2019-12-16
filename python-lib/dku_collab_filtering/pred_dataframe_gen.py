import pandas as pd

# def create_ratings_df(preds, user_id_col, item_id_col):
#     # clean code up
#     item_ids = []
#     user_ids = []
#     act_ratings = []
#     est_ratings = []

#     for row in preds:
#         item_ids.append(row.iid)
#         user_ids.append(row.uid)
#         act_ratings.append(row.r_ui)
#         est_ratings.append(row.est)

#     preds_df = pd.DataFrame(columns=[user_id_col, item_id_col, 'actual_rr', 'pred_rr'])
#     preds_df[user_id_col] = user_ids
#     preds_df[item_id_col] = item_ids
#     preds_df['actual_rating'] = act_ratings
#     preds_df['pred_rating'] = est_ratings
#     return preds_df

def create_ratings_df(preds, user_id_col, item_id_col):
    ratings_df_data = []
    for row in preds:
        row_dict = {}
        row_dict[item_id_col] = row.iid
        row_dict[user_id_col] = row.uid
        row_dict['actual_rating'] = row.r_ui
        row_dict['pred_rating'] = row.est
        ratings_df_data.append(row_dict)
    preds_df = pd.DataFrame(ratings_df_data, columns=[user_id_col, item_id_col, 'actual_rating', 'pred_rating'])
    return preds_df

def get_pivoted_df(master_df, user_id_col, item_id_col, value):
    # input is the best model's predictions, and pivots it to have one row per user and one column per item with user/item predictions filling such columns
    # master_df = create_ratings_df(surprise_algo_preds, user_id_col, item_id_col)
    pivoted_df = master_df.pivot(index=user_id_col, columns=item_id_col, values=value)
    # actual_df_pivoted = master_df.pivot(index=user_id_col, columns=item_id_col, values='')
    return pivoted_df

def merge_preds_actual_dfs(pred_df, actual_df):
    pred_df.columns = [str(col) + '_pred' for col in pred_df.columns]
    actual_df.columns = [str(col) + '_actual' for col in actual_df.columns]
    full_df = pd.merge(pred_df, actual_df, left_index=True, right_index=True)
    return full_df