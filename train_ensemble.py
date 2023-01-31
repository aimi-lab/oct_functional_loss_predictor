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

models_macula = [
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-180848__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_RNFL',
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-181055__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_GCL',
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-181216__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_INL',
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-181313__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_ONL',
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-181432__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_PR',
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-181505__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_CC',
    '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-181550__ep100_bs032_lr1.00E-02_MD_THICK_ADAM_RESNET34_FINAL_RT'
]

model_onh = '/storage/homefs/ds21n601/perimetry_project/resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-123227__ep100_bs032_lr1.00E-02_MD_ONH_ADAM_RNFL_RESNET18'


def validate_after_concat(df):

    drop_cols = []

    # exam uuid are equal
    for ii in range(0, 18, 3):
        assert df.iloc[:, ii].equals(df.iloc[:, ii+3])
        drop_cols.append(ii)

    # true values are equal
    for ii in range(1, 19, 3):
        assert df.iloc[:, ii].equals(df.iloc[:, ii+3])
        drop_cols.append(ii)

    return df.drop(df.columns[drop_cols], axis=1, inplace=False)


def set_df_naming(df):

    df = df.set_index(18)
    rename_dict = {k: v for k, v in zip(range(2, 23, 3), [f'model_{i}' for i in range(7)])}
    rename_dict[19] = 'trues'
    return df.rename(columns=rename_dict, index={18: 'uuid'})


def get_data(dtype, with_onh=True):

    df = pd.concat(map(pd.read_csv, [os.path.join(model, f'{dtype}_predictions.csv') for model in models_macula]), ignore_index=True, axis=1)
    df = validate_after_concat(df)
    df = set_df_naming(df)

    if with_onh:
        df_onh = pd.read_csv(os.path.join(model_onh, f'{dtype}_predictions.csv'))
        df_onh = df_onh.set_index(df_onh.columns[0])

        df_merged = df.merge(df_onh, left_index=True, right_index=True, validate='one_to_one')
        assert df_merged['trues_x'].equals(df_merged['trues_y'])
        df_merged = df_merged.drop(['trues_x'], axis=1).rename(columns={'trues_y': 'trues', 'preds': 'model_onh'})
    else:
        df_merged = df

    return df_merged.drop(['trues'], axis=1), df_merged['trues']


with_onh = True

X_train, y_train = get_data('train', with_onh=with_onh)
X_valid, y_valid = get_data('validation', with_onh=with_onh)
X_test, y_test = get_data('test', with_onh=with_onh)

regr = make_pipeline(StandardScaler(), LinearRegression())
regr.fit(X_valid, y_valid)
y_test_pred = regr.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

text = f'MAE$_{{test}}$: {mae:.2f}\n'
text += f'$R^2_{{test}}$: {r2:.2f}'

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
path_str = os.path.join('resnet', 'runs', f"{current_time}_ENSEMBLE")
if with_onh: path_str += '_withONH'
os.mkdir(path_str)

pd.DataFrame({'preds': y_test_pred, 'trues': y_test}, index=y_test.index).to_csv(
    os.path.join(path_str, 'test_preds.csv')
    )

fig, ax = plt.subplots(figsize=(6, 6))
_plot_truth_pred(ax, y_test, y_test_pred, text=text)
fig.tight_layout()
fig.savefig(os.path.join(path_str, 'true_predictions_plot_only_test.png'))
fig.clf()
plt.close()

print('ciao')