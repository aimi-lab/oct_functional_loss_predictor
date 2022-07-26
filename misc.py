from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, RandomizedSearchCV, validation_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import random
import glob

from constants import RNDM_STATE, G_POINTS, G_CLUSTERS, FEATURES, RETINAL_LAYERS, CIRCLE_RETINAL_LAYERS, GLAUCOMA_GS_THRESHOLDS

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

sns.set_context('paper')

# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
# https://www.kaggle.com/reighns/understanding-the-quadratic-weighted-kappa
# https://www.kaggle.com/c/diabetic-retinopathy-detection/overview/evaluation

OCTOPUS_FILENAME = "001_eyesuite_export_onlyG_onlygt59.csv"
CATCHTRIALS_FILENAME = "001_eyesuite_export_catch_trials_cleaned.csv"
HEYEX_FILENAME = "005_heyex_export_final.csv"
DISCOVERY_FILENAME = "discovery-export-d97ac189-584b-49af-9c82-2afe9487a61a-2022_07_07_19_59.csv"
DISCOVERY_FILENAME_CIRCLES = "discovery-export-28fbddce-9467-4fbd-b38d-2191610204dc-2022_07_25_14_26.csv"

OPS = {'p': np.add, 't': np.multiply, 'd': np.divide, 'm': np.subtract}


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
    return pd.cut(deviation_values, GLAUCOMA_GS_THRESHOLDS, right=True, labels=range(6))

def read_dataset():
    
    indexes = ['Patient ID', 'Eye', 'vf_date']
    df_cv = pd.read_csv(os.path.join('inputs', 'dataset', 'crossval.csv')).set_index(indexes)
    df_test = pd.read_csv(os.path.join('inputs', 'dataset', 'test.csv')).set_index(indexes)
    
    return df_cv, df_test
    
def make_dataset(plot=False):

    ################### HEYEX ###########################

    # df_heyex = pd.read_excel("./inputs/20211206_perimetry_dicom_export_Serife.xlsx")
    df_heyex = pd.read_csv(f"./inputs/{HEYEX_FILENAME}")
    # df_heyex['patient_id'] = df_heyex['patient_id'].fillna(method='ffill').astype(int)
    # df_heyex = df_heyex[df_heyex['uploaded'] == 1.0] # only dicoms uploaded to Discovery
    df_heyex['filename'] = df_heyex['dcm_path'].apply(lambda path: os.path.split(path)[1])
    df_heyex = df_heyex.set_index('filename')
    df_heyex['oct_date'] = pd.to_datetime(df_heyex.dcm_acquisition_date).dt.date
    df_heyex = df_heyex[['patient_id', 'image_laterality', 'oct_date', 'rows', 'columns', 'slices']]
    df_heyex['patient_id'] = df_heyex['patient_id'].astype(str).str.lstrip('0')

    ################### DISCOVERY ###########################

    df_discovery = pd.read_csv(f"./inputs/{DISCOVERY_FILENAME}")

    index_list = ["FILENAME", "STUDYID", "USUBJID", "FOCID", "OEDTC", "OETESTCD"]
    df_discovery = df_discovery[index_list + FEATURES]
    df_discovery = df_discovery[df_discovery['OETESTCD'].isin(RETINAL_LAYERS)] 

    df_discovery.drop_duplicates(['FILENAME', 'OETESTCD'], keep='first', inplace=True) # export yielded undexpected duplicates
    df_discovery = df_discovery.set_index(index_list).unstack()
    df_discovery.columns.names = (None, None)
    # reset MultiIndex in columns with list comprehension
    df_discovery.columns = ['_'.join(col[::-1]).strip('_') for col in df_discovery.columns]

    srf_cols = [col for col in df_discovery.columns if "SRF_" in col]
    df_discovery[srf_cols] = df_discovery[srf_cols].fillna(value=0)

    df_discovery['FILENAME'] = df_discovery.index.get_level_values(0)
    df_discovery = df_discovery.droplevel(0)

    ################## DISCOVERY CIRCLES ######################

    df_circles = pd.read_csv(f"./inputs/{DISCOVERY_FILENAME_CIRCLES}")

    df_circles = df_circles[index_list + ["THICKNESS_BG"]]
    df_circles = df_circles.rename(columns={'THICKNESS_BG': 'THICKNESS_ONH'})
    df_circles = df_circles[df_circles['OETESTCD'].isin(CIRCLE_RETINAL_LAYERS)] 

    df_circles.drop_duplicates(['FILENAME', 'OETESTCD'], keep='first', inplace=True) # export yielded undexpected duplicates
    df_circles = df_circles.set_index(index_list).unstack()
    df_circles.columns.names = (None, None)
    # reset MultiIndex in columns with list comprehension
    df_circles.columns = ['_'.join(col[::-1]).strip('_') for col in df_circles.columns]
    df_circles = df_circles.droplevel(0)

    df_discovery = df_discovery.merge(df_circles, left_index=True, right_index=True, how='inner', validate='one_to_one')
    df_discovery = df_discovery.set_index("FILENAME")

    df_discovery = df_discovery[df_discovery['RNFL_THICKNESS_ONH'] > 50]

    ################### EYESUITE ###########################

    df_octopus = pd.read_csv(os.path.join("inputs", OCTOPUS_FILENAME))
    df_octopus['Examination'] = pd.to_datetime(df_octopus.Examination)
    df_octopus['Date of birth'] = pd.to_datetime(df_octopus['Date of birth'])
    df_octopus['Age'] = (df_octopus.Examination - df_octopus['Date of birth']).astype('<m8[Y]')
    df_octopus['vf_date'] = df_octopus.Examination.dt.date
    df_octopus = df_octopus.sort_values('Examination', ascending=False)
    df_octopus.drop_duplicates(['Patient ID', 'Eye', 'vf_date'], inplace=True, keep='first')
    df_octopus['Patient ID'] = df_octopus['Patient ID'].astype(str).str.lstrip('0')

    df_catchtrials = pd.read_csv(os.path.join("inputs", CATCHTRIALS_FILENAME))
    df_catchtrials['Examination'] = pd.to_datetime(df_catchtrials.Examination)
    df_catchtrials['vf_date'] = df_catchtrials.Examination.dt.date
    df_catchtrials = df_catchtrials.sort_values('Examination', ascending=False)
    df_catchtrials.drop_duplicates(['Patient ID', 'Eye', 'vf_date'], inplace=True, keep='first')
    df_catchtrials['Patient ID'] = df_catchtrials['Patient ID'].astype(str).str.lstrip('0')
    df_catchtrials = df_catchtrials[['Patient ID', 'Eye', 'vf_date', '{FALSEPOSITIFCATCHTRIAL}']]
    df_octopus = df_octopus.merge(df_catchtrials, on=['Patient ID', 'Eye', 'vf_date'], validate='one_to_one', how='left')

    # drop non-valid exams
    df_octopus = df_octopus[df_octopus['Age'] >= 40]
    df_octopus = df_octopus[df_octopus['Age'] <= 95]
    df_octopus = df_octopus[df_octopus['{FALSENEGATIFCATCHTRIAL}'] < 0.15 * df_octopus['{NEGATIFCATCHTRIAL}']]
    df_octopus = df_octopus[df_octopus['{FALSEPOSITIFCATCHTRIAL}'] < 0.15 * df_octopus['{POSITIFCATCHTRIAL}']]

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
        sns.histplot(data=df_merged, x='Age', bins=range(40, 101))
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
    df_ML.to_csv(os.path.join(dataset_dir, 'full.csv'))

def _create_feature(df, layers, op, sector):
    start_layer = True
    for layer in layers:
        if start_layer:
            result = df[layer + '_' + sector].values
            start_layer = False
        else:
            result = OPS[op](result, df[layer + '_' + sector].values)
    col_name = op.join(layers) + '_' + sector
    df[col_name] = result
    df[col_name] = df[col_name].fillna(0)
    df[col_name] = df[col_name].replace([np.inf, -np.inf], 1E6)

def _detect_operation(layers_str):

    for operation in OPS.keys():
        if operation in layers_str:
            return operation

def augment_features(*dfs, n=200):
    
    sector_pool = [feat for feat in FEATURES if "THICKNESS" in feat] + ["THICKNESS_ONH"]
    layers_pool = [lay for lay in RETINAL_LAYERS if lay not in ['IRF', 'SRF', 'PED']]
    target_n_cols = len(dfs[0].columns) + n

    while len(dfs[0].columns) < target_n_cols:
        n_ops = random.randint(1, 5)
        op = random.choice(list(OPS.keys()))

        # limit operations to keep values far from 0 and Inf
        if op in ['t', 'd']:
            n_ops = 1

        layers = random.sample(layers_pool, n_ops + 1) # layers are one more than ops

        # if op is commutative, sort layers as it will have same values but different col_name
        # if op is not commutative, sort only layers starting from second
        if op in ['t', 'p']:
            layers.sort()
        else:
            layers[1:] = sorted(layers[1:])

        for sector in sector_pool:
            for df in dfs:
                _create_feature(df, layers, op, sector)

    cols_dfs = [len(df.columns) for df in dfs]
    assert len(set(cols_dfs)) == 1, 'Feature augmentation failed'
    return dfs

def create_features(from_file, *dfs):

    print('Creating features read from ' + from_file)
    delete_cols = list(dfs[0].columns)

    with open(from_file, 'r') as f:
        features = f.readlines()
        features = [line.rstrip() for line in features]

    for feature in features:
        if feature in delete_cols:
            delete_cols.remove(feature)
            continue

        layers_str, *sector_list = feature.split('_')
        op = _detect_operation(layers_str)
        layers = layers_str.split(op)
        sector = '_'.join(sector_list)

        for df in dfs:
            _create_feature(df, layers, op, sector)
    
    for df in dfs:
        df.drop(columns=delete_cols, inplace=True)

    return dfs

def analyse_augmented_features(dir):
    feat_dict = dict()

    files = glob.glob(os.path.join(dir, '*_important_features.txt'))

    for afile in files:
        with open(afile, 'r') as f:
            features = f.readlines()
            features = [line.rstrip() for line in features]

        for i, feature in enumerate(features):

            layers, *suffix = feature.split('_')
            
            # if division, handle reciprocal features, ex: RNFLdGCL is reciprocal as GCLdRNFL
            if 'd' in layers:
                layers_list = layers.split('d')
                layers_list.sort()
                layers = 'd'.join(layers_list)
                feature = '_'.join([layers] + suffix)

            feat_dict[feature] = max(feat_dict.get(feature, 0), 40 - i)

    out_list = sorted(feat_dict.keys(), key=lambda x: x[1], reverse=True)[:200]

    with open(os.path.join(dir, '00_important_features.txt'), 'w') as fp:
        fp.write('\n'.join(out_list))

def run_grid_search(X, y, model, cv_splitter, cv_grid, scoring='neg_mean_absolute_error', sample_weights=None, random=False):
    
    print('Running Grid Search to optimise ' + scoring)
    if random:
        # FIXME: adjust number of random iterations
        grid_search = RandomizedSearchCV(model, cv_grid, cv=cv_splitter, scoring=scoring, verbose=2, n_jobs=-1, n_iter=100, random_state=RNDM_STATE)
    else:
        grid_search = GridSearchCV(model, cv_grid, cv=cv_splitter, scoring=scoring, verbose=2, n_jobs=-1)

    if sample_weights is not None:
        print(sample_weights) 
        grid_search.fit(X, y, sample_weight=sample_weights)
    else:
        grid_search.fit(X, y)

    print(f'Best parameters from grid search: {grid_search.best_params_}')

    return grid_search.best_estimator_

if __name__ == '__main__':

    # from plotting_utils import _plot_etdrs_grids

    # dfs = read_dataset()
    # # df = pd.concat(dfs, ignore_index=False)
    # df = dfs[1]
    # df = df.filter(regex='THICKNESS')
    # col2drop = [col for col in df.columns if col.startswith('RT')]
    # # print(col2drop)
    # df = df.drop(col2drop, axis=1)
    
    # interesting_cases = [(2436604, 'OD', '2012-01-04'),
    #     (1925970, 'OD', '2012-07-04'),
    #     (3360652, 'OS', '2017-01-13'),
    #     (14648261, 'OS', '2019-07-19'),
    #     (1083775, 'OS', '2018-02-16'),
    #     (2095602, 'OD', '2020-11-13')]
    
    # for row in df.iterrows():
    #     if row[0] in interesting_cases:
    #         _plot_etdrs_grids(row[1], './inputs/etdrs_images', filename='_'.join([str(i) for i in row[0]])+'.png', max=150)

    #####################

    # analyse_augmented_features("/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD")

    #####################

    make_dataset(plot=True)

    #####################

    # from sklearn.manifold import TSNE
    # from sklearn.preprocessing import StandardScaler
    # import umap

    # df_full = pd.concat(read_dataset())

    # features = df_full.filter(regex='THICKNESS|VOLUME')
    # features_scaled = StandardScaler().fit_transform(features)
    # # features_embedded = TSNE(learning_rate='auto', init='pca', perplexity=50, early_exaggeration=30).fit_transform(features_scaled)
    # tsne_embedding = TSNE(perplexity=100, early_exaggeration=70).fit_transform(features_scaled)
    # umap_embedding = umap.UMAP().fit_transform(features_scaled)

    # fig, ax = plt.subplots()
    # sns.scatterplot(x=tsne_embedding[:, 0], y=tsne_embedding[:, 1], hue=df_full["GS"], ax=ax, palette='Spectral')
    # fig.savefig('TSNE_embedding.png')

    # fig, ax = plt.subplots()
    # sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], hue=df_full["GS"], ax=ax, palette='Spectral')
    # fig.savefig('UMAP_embedding.png')

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