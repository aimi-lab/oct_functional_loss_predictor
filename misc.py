from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import random

from constants import RNDM_STATE, G_POINTS, G_CLUSTERS, FEATURES, RETINAL_LAYERS

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

sns.set_context('paper')

# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
# https://www.kaggle.com/reighns/understanding-the-quadratic-weighted-kappa
# https://www.kaggle.com/c/diabetic-retinopathy-detection/overview/evaluation

OCTOPUS_FILENAME = "001_eyesuite_export_onlyG_onlygt59.csv"
HEYEX_FILENAME = "005_heyex_export_final.csv"
DISCOVERY_FILENAME = "discovery-export-d97ac189-584b-49af-9c82-2afe9487a61a-2022_07_07_19_59.csv"

def create_dir_if_not_exist(dir_name, prefix='outputs'):
    
    dir_path = os.path.join(prefix, dir_name) if prefix else dir_name

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path

def create_overlaid_images():
    fig, ax = plt.subplots()

    # clip_circle = plt.Circle((0, 0), 298, fill=False)
    # ax.add_patch(clip_circle)
    # img1 = plt.imread(os.path.join('inputs', 'g_pattern.png'))
    # ax.imshow(img1[:, ::-1], extent=[-300, 300, -300, 300], clip_path=clip_circle, clip_on=True)

    img1 = plt.imread(os.path.join('inputs', 'g_clusters.png'))
    ax.imshow(img1, extent=[-285, 285, -285, 285])
    img2 = plt.imread(os.path.join('inputs', 'fovea_top_view.png'))
    ax.imshow(img2, extent=[-140, 140, -140, 140], alpha=0.5)
    circle1 = plt.Circle((0, 0), 108, color='grey', fill=False)
    circle2 = plt.Circle((0, 0), 108/2, color='grey', fill=False)
    circle3 = plt.Circle((0, 0), 108/6, color='grey', fill=False)

    ax.plot((np.cos(math.pi/4)*108/6, np.cos(math.pi/4)*108), (np.sin(math.pi/4)*108/6, np.sin(math.pi/4)*108), color='grey', linewidth=2.4)
    ax.plot((np.cos(-math.pi/4)*108/6, np.cos(-math.pi/4)*108), (np.sin(-math.pi/4)*108/6, np.sin(-math.pi/4)*108), color='grey', linewidth=2.4)
    ax.plot((np.cos(3*math.pi/4)*108/6, np.cos(3*math.pi/4)*108), (np.sin(3*math.pi/4)*108/6, np.sin(3*math.pi/4)*108), color='grey', linewidth=2.4)
    ax.plot((np.cos(5*math.pi/4)*108/6, np.cos(5*math.pi/4)*108), (np.sin(5*math.pi/4)*108/6, np.sin(5*math.pi/4)*108), color='grey', linewidth=2.4)

    # ax.set_aspect('equal')
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    # plt.xticks(np.arange(-300, 300, step=50))
    # plt.yticks(np.arange(-300, 300, step=50))
    plt.axis('off')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig('GGG.png')
    fig.clf()
    plt.close()

def classify_glaucoma(deviation_values):
    return pd.cut(deviation_values, [-20, -0.8, 4.4, 9.5, 15.3, 23.1, 50], right=True, labels=range(6))

def read_dataset():
    
    indexes = ['Patient ID', 'Eye', 'vf_date']
    df_cv = pd.read_csv(os.path.join('inputs', 'dataset', 'crossval.csv')).set_index(indexes)
    df_test = pd.read_csv(os.path.join('inputs', 'dataset', 'test.csv')).set_index(indexes)
    
    return df_cv, df_test
    
def make_dataset(plot=False):

    # df_heyex = pd.read_excel("./inputs/20211206_perimetry_dicom_export_Serife.xlsx")
    df_heyex = pd.read_csv(f"./inputs/{HEYEX_FILENAME}")
    # df_heyex['patient_id'] = df_heyex['patient_id'].fillna(method='ffill').astype(int)
    # df_heyex = df_heyex[df_heyex['uploaded'] == 1.0] # only dicoms uploaded to Discovery
    df_heyex['filename'] = df_heyex['dcm_path'].apply(lambda path: os.path.split(path)[1])
    df_heyex = df_heyex.set_index('filename')
    df_heyex['oct_date'] = pd.to_datetime(df_heyex.dcm_acquisition_date).dt.date
    df_heyex = df_heyex[['patient_id', 'image_laterality', 'oct_date', 'rows', 'columns', 'slices']]
    df_heyex['patient_id'] = df_heyex['patient_id'].astype(str).str.lstrip('0')

    df_discovery = pd.read_csv(f"./inputs/{DISCOVERY_FILENAME}")

    # if plot:
    #     fig, ax = plt.subplots()
    #     total = df_heyex.slices.value_counts().sum()
    #     df_heyex.slices.astype(int).value_counts().plot(kind='pie', autopct=lambda p: '{:.0f}%\n{:.0f}'.format(p, p * total / 100) if p > 5 else '', ylabel='', colormap='Greens')
    #     fig.savefig('pie_plot_initial.png')
    #     fig.clf()
    #     plt.close() 

    # # df_discovery = df_discovery[~df_discovery.PIXL2.isin([3, 24, 73])] # get rid of ONH scans!!
    # # df_discovery = df_discovery[~df_discovery.FILENAME.isin(BAD_DICOMS)] # badly located scan
    # # df_discovery = df_discovery[~(df_discovery.PIXL2.eq(49) & df_discovery.PIXL1.eq(768))] # Volumes with 49 slices, but not on 6x6x1.9 mm grid

    # if plot:
    #     fig, ax = plt.subplots()
    #     filt = df_discovery.OESEQ == 1
    #     total = df_discovery[filt].PIXL2.value_counts().sum()
    #     df_discovery[filt].PIXL2.astype(int).value_counts().plot(kind='pie', autopct=lambda p: '{:.0f}%\n{:.0f}'.format(p, p * total / 100) if p > 5 else '', ylabel='', colormap='Greens')
    #     fig.savefig('pie_plot_final.png')
    #     fig.clf()
    #     plt.close()

    df_discovery = df_discovery[["FILENAME", "OETESTCD"] + FEATURES]
    df_discovery = df_discovery[df_discovery['OETESTCD'].isin(RETINAL_LAYERS)] 

    df_discovery.drop_duplicates(['FILENAME', 'OETESTCD'], keep='first', inplace=True) # export yielded undexpected duplicates
    df_discovery = df_discovery.set_index(['FILENAME', 'OETESTCD']).unstack()
    df_discovery.columns.names = (None, None)
    # reset MultiIndex in columns with list comprehension
    df_discovery.columns = ['_'.join(col[::-1]).strip('_') for col in df_discovery.columns]

    srf_cols = [col for col in df_discovery.columns if "SRF_" in col]
    df_discovery[srf_cols] = df_discovery[srf_cols].fillna(value=0)

    df_octopus = pd.read_csv(os.path.join("inputs", OCTOPUS_FILENAME))
    df_octopus['Examination'] = pd.to_datetime(df_octopus.Examination)
    df_octopus['Date of birth'] = pd.to_datetime(df_octopus['Date of birth'])
    df_octopus['Age'] = (df_octopus.Examination - df_octopus['Date of birth']).astype('<m8[Y]')
    df_octopus['vf_date'] = df_octopus.Examination.dt.date
    df_octopus = df_octopus.sort_values('Examination', ascending=False)
    df_octopus.drop_duplicates(['Patient ID', 'Eye', 'vf_date'], inplace=True, keep='first')
    df_octopus['Patient ID'] = df_octopus['Patient ID'].astype(str).str.lstrip('0')

    # drop non-valid exams
    df_octopus = df_octopus[df_octopus[' {FALSENEGATIFCATCHTRIAL}'] < 0.2 * df_octopus['{NEGATIFCATCHTRIAL}']]

    ## EXTRACT CLUSTERS ##

    idx = df_octopus.columns.tolist().index('{POSXYPH1PH2NV}') # get column index and use it to select columns up to 365 after

    perimetry_values = df_octopus[df_octopus.columns[idx:idx+G_POINTS*5]].to_numpy().reshape(-1, 5)
    patients_idxs = np.repeat(df_octopus['Patient ID'].to_numpy(), G_POINTS)
    eye_idxs = np.repeat(df_octopus['Eye'].to_numpy(), G_POINTS)
    dates_idxs = np.repeat(df_octopus['vf_date'].to_numpy(), G_POINTS)
    n_points = np.repeat(df_octopus['{TESTLOCENUMBER}'].to_numpy(), G_POINTS)

    df_cluster = pd.DataFrame(np.hstack((
        np.expand_dims(patients_idxs, axis=-1), 
        np.expand_dims(eye_idxs, axis=-1), 
        np.expand_dims(dates_idxs, axis=-1), 
        np.expand_dims(n_points, axis=-1), 
        perimetry_values
        )),
        columns=['Patient ID', 'Eye', 'vf_date', 'n_points', 'X', 'Y', 'Norm', 'Dummy', 'Meas'])

    df_cluster['X'] = df_cluster['X'].astype(float)
    df_cluster['Y'] = df_cluster['Y'].astype(float)
    df_cluster.loc[df_cluster['Eye'] == 'OD', 'X'] = - df_cluster.loc[df_cluster['Eye'] == 'OD', 'X'].values # swap based on laterality

    df_cluster = df_cluster.drop('Dummy', axis=1)
    df_cluster = df_cluster.drop(df_cluster[df_cluster.Meas == -1].index)
    df_cluster['Defect'] = df_cluster['Meas'] - df_cluster['Norm'] 
    df_cluster['Cluster'] = df_cluster[['X', 'Y']].apply(tuple, axis=1).map(G_CLUSTERS)
    df_cluster.dropna(subset=['Cluster'], inplace=True)

    # df_cluster = df_cluster[df_cluster.groupby(['Patient ID', 'Eye', 'VF Date']).Cluster.transform('count') == 58]
    assert int(df_cluster.groupby(['Patient ID', 'Eye', 'vf_date']).count().Cluster.unique()) == 58, 'Not all exams have 59 points detected in grid'

    df_cluster = df_cluster.groupby(['Patient ID', 'Eye', 'vf_date', 'Cluster'])['Defect'].mean() / 10
    df_cluster = df_cluster.unstack()

    ######################

    df_octopus = df_octopus[['Patient ID', 'Age', 'Eye', 'vf_date', '{MD}', '{MS}']] \
        .rename(columns={'{MD}': 'MD', '{MS}': 'MS'}) \
        .set_index(['Patient ID', 'Eye', 'vf_date'])

    df_merged = df_discovery.merge(df_heyex, left_index=True, right_index=True)
    df_merged['image_laterality'] = df_merged['image_laterality'].replace({'L': 'OS', 'R': 'OD'})
    df_merged = df_merged.set_index(['patient_id', 'image_laterality', 'oct_date'])
    
    df_merged.index.names = df_octopus.index.names
    df_merged = df_merged.merge(df_octopus, left_index=True, right_index=True)
    df_merged = df_merged.merge(df_cluster, left_index=True, right_index=True, validate='one_to_one', how='left')

    if plot:
        fig, ax = plt.subplots()
        sns.histplot(x=df_merged.index.get_level_values(1))
        ax.set_xlabel('Laterality')
        fig.savefig('histplot_laterality.png', bbox_inches='tight')
        fig.clf()
        plt.close()

        fig, ax = plt.subplots()
        sns.histplot(data=df_merged, x='Age')
        ax.set_xlabel('Age [yr]')
        fig.savefig('histplot_age.png', bbox_inches='tight')
        fig.clf()
        plt.close()

        fig, ax = plt.subplots()
        sns.histplot(data=df_merged, y='MD')
        ax.set_ylabel('MD [dB]')
        fig.savefig('histplot_md.png', bbox_inches='tight')
        fig.clf()
        plt.close()

        fig, ax = plt.subplots()
        sns.scatterplot(data=df_merged, x='Age', y='MD')
        ax.set_xlabel('Age [yr]')
        ax.set_ylabel('MD [dB]')
        fig.savefig('plot_age_md.png', bbox_inches='tight')
        fig.clf()
        plt.close()

    df_ML = df_merged.drop(['rows', 'columns'], axis=1) # remove "index", put "Patient ID" and "Eye"
    df_ML['GS'] = classify_glaucoma(df_ML['MD'])
    df_ML = df_ML[df_ML['GS'] < 5]

    # features, target, glauc_stage = df_ML.drop(['MD', 'MS'], axis=1), df_ML[target], classify_glaucoma(df_ML['MD'])

    df_ML = df_ML.sample(frac=1)
    # features, target, glauc_stage, extras = shuffle(features, target, glauc_stage, extras, random_state=RNDM_STATE)

    # if return_unsplit: return features, target, glauc_stage, extras, None, None, None, None
    
    cv_idxs, test_idxs = list(StratifiedGroupKFold(n_splits=10).split(df_ML, df_ML['GS'], groups=df_ML.index.get_level_values(0)))[1]

    df_cv = df_ML.iloc[cv_idxs]
    df_test = df_ML.iloc[test_idxs]

    dataset_dir = create_dir_if_not_exist('dataset', 'inputs')

    df_cv.to_csv(os.path.join(dataset_dir, 'crossval.csv'))
    df_test.to_csv(os.path.join(dataset_dir, 'test.csv'))
    
def augment_features(*dfs, n=200):
    
    sector_pool = [feat for feat in FEATURES if "THICKNESS" in feat]
    layers_pool = [lay for lay in RETINAL_LAYERS if lay not in ['SRF', 'PED']]
    target_n_cols = len(dfs[0].columns) + n

    ops = {'p': np.add, 't': np.multiply, 'd': np.divide, 'm': np.subtract}

    while len(dfs[0].columns) < target_n_cols:
        n_ops = random.randint(1, 5)
        op = random.choice(list(ops.keys()))

        # limit operations to keep values far from 0 and Inf
        if op in ['t', 'd']:
            n_ops = 1

        layers = random.sample(layers_pool, n_ops + 1) # layers are one more than ops

        # if op is commutative, sort layers as it will have same values but different col_name
        # if op is not commutative, sort only layers from second
        if op in ['t', 'p']:
            layers.sort()
        else:
            layers[1:] = sorted(layers[1:])

        for sector in sector_pool:
            for df in dfs:
                start_layer = True
                for layer in layers:
                    if start_layer:
                        result = df[layer + '_' + sector].values
                        start_layer = False
                    else:
                        result = ops[op](result, df[layer + '_' + sector].values)
                col_name = op.join(layers) + '_' + sector
                df[col_name] = result
                df[col_name] = df[col_name].fillna(0)
                df[col_name] = df[col_name].replace([np.inf, -np.inf], 1E6)

    cols_dfs = [len(df.columns) for df in dfs]
    assert len(set(cols_dfs)) == 1, 'Feature augmentation failed'
    return dfs


def run_grid_search(X, y, model, cv_splitter, cv_grid, scoring='neg_mean_absolute_error'):
    
    grid_search = GridSearchCV(model, cv_grid, cv=cv_splitter, scoring=scoring, verbose=2) # verbose=3
    grid_search.fit(X, y)

    print(f'Best parameters from grid search: {grid_search.best_params_}')

    return grid_search.best_estimator_

if __name__ == '__main__':

    make_dataset(plot=True)

    #####################

    # from sklearn.manifold import TSNE
    # from sklearn.preprocessing import StandardScaler

    # df_full = pd.concat(read_dataset())

    # features = df_full.filter(regex='THICKNESS|VOLUME')
    # features_scaled = StandardScaler().fit_transform(features)
    # # features_embedded = TSNE(learning_rate='auto', init='pca', perplexity=50, early_exaggeration=30).fit_transform(features_scaled)
    # features_embedded = TSNE(perplexity=100, early_exaggeration=70).fit_transform(features_scaled)

    # fig, ax = plt.subplots()
    # sns.scatterplot(x=features_embedded[:, 0], y=features_embedded[:, 1], hue=df_full["GS"], ax=ax, palette='Reds')
    # fig.savefig('TSNE_dataset.png')

    #####################

    # get_g_clusters()
    # features, md, gs, extras, *others = read_process_split_data('.', FEATURES, RETINAL_LAYERS, False, 'MD', plot=False)

    # # print(gs)
    # df = features.filter(regex='^(RNFL|GCL\+IPL)\_THICKNESS', axis=1)
    # df = df.stack().reset_index(level=-1).rename(columns={'level_2': 'feature', 0: 'value'})

    # df = df.join(gs).rename(columns={'MD': 'GS'})
    # df_rnfl = df[df.feature.str.contains('RNFL')]
    # df_gcl = df[df.feature.str.contains('GCL')]
    # # print(df_gcl.head(30))

    # fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 10), sharex=True)
    # sns.boxplot(x='feature', y='value', hue='GS', data=df_rnfl, ax=ax1)
    # sns.boxplot(x='feature', y='value', hue='GS', data=df_gcl, ax=ax2)

    # ax1.set_xlabel('')
    # ax2.set_xlabel('Grid Location')
    # ax1.set_ylabel(r'Thickness [$\mu$m]')
    # ax2.set_ylabel(r'Thickness [$\mu$m]')
    # ax1.set_title('RNFL Thickness')
    # ax2.set_title('GCL+IPL Thickness')
    # ax1.get_legend().remove()
    # ax2.get_legend().remove()
    # ax2.set_xticklabels(labels=[str(lbl).split('_')[-1][:-2] for lbl in ax2.get_xticklabels()])
    # # ax2.legend(loc='center left', bbox_to_anchor=(1, 1), title='GS')
    # handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', title='Glaucoma Stage')

    # fig.tight_layout()
    # fig.savefig('bbb.png')