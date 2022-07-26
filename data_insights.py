import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from constants import RETINAL_LAYERS

sns.set_style('white')
# sns.set_context('poster')

df = pd.read_csv('./inputs/dataset/full.csv')
# print(df.head())

for lay in RETINAL_LAYERS:
    cols = [col for col in df.columns if col.split('_')[0] == lay and 'THICKNESS' in col]
    df_layer = pd.melt(df, id_vars=['GS'], value_vars=cols)
    # df_layer = df[cols].stack()
    # print(df_layer)
    # df_layer['GS'] = np.repeat(df['GS'].values, len(cols))

    print(df_layer.head())

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x='variable', y='value', hue='GS', data=df_layer, ax=ax, showmeans=False, showfliers=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='GS')
    # print(ax.get_xticklabels())
    ax.set_xticklabels(['_'.join(str(lbl).split('_')[1:])[:-2] for lbl in ax.get_xticklabels()], rotation=90)
    fig.tight_layout()
    ax.set_xlabel(lay)
    ax.set_ylabel('Thickness [$\mu$m]')
    fig.savefig(f'./outputs/insights/boxplots/boxplot_{lay}.png')

for lay in RETINAL_LAYERS:
    cols = [col for col in df.columns if col.split('_')[0] == lay and 'THICKNESS' in col]

    df_adim = pd.DataFrame()
    df_adim['GS'] = df['GS']
    for col in cols:
        df_adim[col] = df[col] / df[col.replace(lay, 'RT')]

    df_layer = pd.melt(df_adim, id_vars=['GS'], value_vars=cols)
    # df_layer = df[cols].stack()
    # print(df_layer)
    # df_layer['GS'] = np.repeat(df['GS'].values, len(cols))

    print(df_layer.head())

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x='variable', y='value', hue='GS', data=df_layer, ax=ax, showmeans=False, showfliers=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='GS')
    # print(ax.get_xticklabels())
    ax.set_xticklabels(['_'.join(str(lbl).split('_')[1:])[:-2] for lbl in ax.get_xticklabels()], rotation=90)
    fig.tight_layout()
    ax.set_xlabel(f'{lay}/RT')
    ax.set_ylabel('Thickness adim. [-]')
    fig.savefig(f'./outputs/insights/boxplots/boxplot_{lay}_adim.png')