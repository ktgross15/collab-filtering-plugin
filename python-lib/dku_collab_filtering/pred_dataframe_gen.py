import pandas as pd

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

def get_pivoted_pred_act_dfs(surprise_algo_preds, u_id_col, i_id_col):
    # input is the best model's predictions, and pivots it to have one row per user and one column per item with user/item predictions filling such columns
    master_df = create_ratings_df(surprise_algo_preds, u_id_col, i_id_col)
    predictions_df_pivoted = master_df.pivot(index=u_id_col, columns=i_id_col, values='pred_rr')
    actual_df_pivoted = master_df.pivot(index=u_id_col, columns=i_id_col, values='actual_rr')

    return predictions_df_pivoted, actual_df_pivoted

def merge_preds_actual_dfs(pred_df, actual_df):
    pred_df.columns = [str(col) + '_pred' for col in pred_df.columns]
    actual_df.columns = [str(col) + '_actual' for col in actual_df.columns]
    full_df = pd.merge(pred_df, actual_df, left_index=True, right_index=True)
    return full_df