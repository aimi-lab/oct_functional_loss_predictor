import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from resnet.libs.utils import _plot_truth_pred
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

models = [
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-163935__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_RNFL',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-164045__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_GCL',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-164130__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_INL',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-164248__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_ONL',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-164349__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_PR',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-164449__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_CC',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-164652__ep100_bs032_lr1.00E-04_clusters_THICK_ADAM_RESNET50_FINAL_RT',
'/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220807-163657__ep100_bs032_lr1.00E-04_clusters_ONH_ADAM_RESNET50_FINAL'
]

# model_onh = '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220805-123140__ep100_bs032_lr1.00E-02_clusters_ONH_ADAM_RESNET18_FINAL'


# def set_df_naming(df):

#     df = df.set_index(18)
#     rename_dict = {k: v for k, v in zip(range(2, 23, 3), [f'model_{i}' for i in range(7)])}
#     rename_dict[19] = 'trues'
#     return df.rename(columns=rename_dict, index={18: 'uuid'})

def get_data(dtype, clust_no):

    dfs = [pd.read_csv(os.path.join(model, f'{dtype}_predictions.csv'), index_col=0)[[f'preds_Cluster_{clust_no:02d}', f'trues_Cluster_{clust_no:02d}']]
        .rename(columns={f'preds_Cluster_{clust_no:02d}': 'preds', f'trues_Cluster_{clust_no:02d}': 'trues'}) for model in models]
    df = pd.concat(dfs, ignore_index=True, axis=1)

    df_preds = df[[i for i in range(len(models)*2) if i % 2 == 0]]
    df_trues = df[[1]]

    return df_preds, df_trues

with_onh = True

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
folder = f'{current_time}_ENSEMBLE_CLUSTER'
if with_onh: folder += '_withONH'

for clust in range(1, 11):

    X_train, y_train = get_data('train', clust)
    X_valid, y_valid = get_data('validation', clust)
    X_test, y_test = get_data('test', clust)

    regr = make_pipeline(StandardScaler(), LinearRegression())
    regr.fit(X_valid, y_valid)

    y_test_pred = regr.predict(X_test).flatten()
    y_test = y_test.values.flatten()

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    text = f'MAE$_{{test}}$: {mae:.2f}\n'
    text += f'$R^2_{{test}}$: {r2:.2f}'

    path_str = os.path.join(
        'resnet', 
        'runs', 
        folder, 
        f'cluster_{clust:02d}'
        )
    os.makedirs(path_str, exist_ok=True)

    pd.DataFrame({'preds': y_test_pred, 'trues': y_test}, index=X_test.index).to_csv(
        os.path.join(path_str, f'test_values.csv')
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    _plot_truth_pred(ax, y_test, y_test_pred, text=text)
    fig.tight_layout()
    fig.savefig(os.path.join(path_str, f'true_predictions_plot_only_test.png'))
    fig.clf()
    plt.close()

    print('ciao')