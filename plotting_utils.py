import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from sklearn.model_selection import cross_validate, RepeatedKFold
import os
import numpy as np
import seaborn as sns
import itertools
from constants import RETINAL_LAYERS

np.set_printoptions(precision=2)
sns.set_context('poster')

CMAP = cm.get_cmap('Greens', 5)
# CMAP.set_over('k')

N_MESH = 200
ANGLE_DICT = {
    'C': np.linspace(0, 2*np.pi, 4*N_MESH),
    'T': np.linspace(-np.pi/4, np.pi/4, N_MESH),
    'S': np.linspace(np.pi/4, 3*np.pi/4, N_MESH),
    'N': np.linspace(3*np.pi/4, 5*np.pi/4, N_MESH),
    'I': np.linspace(5*np.pi/4, 7*np.pi/4, N_MESH),
    }
RADIUS_DICT = {'1': [0, 1], '3': [1, 3], '6': [3, 6]}
AXES_COLS = {'THICKNESS': 0, 'VOLUME': 1}
AXES_COLS_INV = {v: k for k, v in AXES_COLS.items()}
LW=1


def _plot_etdrs_grids(pd_serie, save_dir):

    # create right number of rows in subplot and map it to dict
    important_layers = set([i.split('_')[0] for i in pd_serie.index])
    axes_rows = {}
    ii = 0
    for ll in RETINAL_LAYERS:
        if ll in important_layers:
            axes_rows[ll] = ii
            ii += 1

    fig, axs = plt.subplots(ii, 2, figsize=(6, 6*ii/2), subplot_kw=dict(projection="polar"))
    norm = colors.Normalize(0, pd_serie.max()) 

    for ax in axs.reshape(-1):
        ax.plot([np.pi/4, np.pi/4], [1, 6], zorder=3, color='k', linewidth=LW)   
        ax.plot([3*np.pi/4, 3*np.pi/4], [1, 6], zorder=3, color='k', linewidth=LW) 
        ax.plot([5*np.pi/4, 5*np.pi/4], [1, 6], zorder=3, color='k', linewidth=LW) 
        ax.plot([-np.pi/4, -np.pi/4], [1, 6], zorder=3, color='k', linewidth=LW) 

        ax.plot(np.linspace(0, 2*np.pi, 365), np.ones(365), zorder=3, color='k', linewidth=LW)    
        ax.plot(np.linspace(0, 2*np.pi, 365), np.ones(365)*3, zorder=3, color='k', linewidth=LW) 
        [x.set_linewidth(LW) for x in ax.spines.values()]   

    for i, v in pd_serie.iteritems():

        if i == 'Age': continue
        layer, vol_thick, [sector, rad] = i.split('_')

        t = ANGLE_DICT[sector] # theta values
        r = RADIUS_DICT[rad] # radius values
        # _, tg = np.meshgrid(r, t) # create a r,theta meshgrid
        c = np.ones((len(r), len(t))) * v # uniform color in sector

        im = axs[axes_rows[layer], AXES_COLS[vol_thick]].pcolormesh(t, r, c[:-1, :-1], shading='flat', cmap=CMAP, norm=norm)  #plot the colormesh on axis with colormap

    for ax in axs.reshape(-1):
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        text_kwargs = dict(ha='center', va='center', fontsize=12, color='black')
        ax.text(0, 4.5, 'T6', **text_kwargs)
        ax.text(np.pi, 4.5, 'N6', **text_kwargs)
        ax.text(np.pi/2, 4.5, 'S6', **text_kwargs)
        ax.text(-np.pi/2, 4.5, 'I6', **text_kwargs)
        ax.text(0, 2, 'T3', **text_kwargs)
        ax.text(np.pi, 2, 'N3', **text_kwargs)
        ax.text(np.pi/2, 2, 'S3', **text_kwargs)
        ax.text(-np.pi/2, 2, 'I3', **text_kwargs)
        ax.text(0, 0, 'C1', **text_kwargs)

    for nn, ax in enumerate(axs[0, :]):
        ax.set_title(AXES_COLS_INV[nn].lower().capitalize())

    axes_rows_inv = {v: k for k, v in axes_rows.items()}
    for nn, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(axes_rows_inv[nn])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.outline.set_linewidth(LW)
    cb.dividers.set_linewidth(LW)

    fig.savefig(os.path.join(save_dir, 'etdrs_grid_importance.png'), bbox_inches='tight')

def plot_feature_importance(X, y, model, cv, save_dir):

    cv_model = cross_validate(
        model,
        X,
        y,
        cv=RepeatedKFold(n_splits=cv, n_repeats=5),
        return_estimator=True
        )
    
    if hasattr(cv_model['estimator'][0], 'feature_importances_'):
        coefs = pd.DataFrame(
            [
                est.feature_importances_ for est in cv_model["estimator"]
            ],
            columns=model.feature_names_in_,
        )
    else:
        coefs = pd.DataFrame(
            [
                est[1].coef_[0] * X.std(axis=0)
                for est in cv_model["estimator"]
            ],
            columns=model.feature_names_in_,
        )     
    
    meds = abs(coefs.median()).sort_values(ascending=False).head(20)
    coefs = coefs[meds.index]

    _plot_etdrs_grids(meds, save_dir)

    fig = plt.figure(figsize=(12, 9))
    sns.stripplot(data=coefs, orient="h", color="k", alpha=0.5)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient importance")
    plt.title("Coefficient importance and its variability")
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'feature_importance.png'))

def plot_truth_prediction(df, save_dir, lim=None, text=None):

    ############## ONLY TEST ########################

    fig, ax = plt.subplots(figsize=(12, 8))

    df_test = df[df.dataset == 'test']
    sns.scatterplot(data=df_test, x="y", y="y_pred", ax=ax) #, style="n_slices", ax=ax)

    # ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
    ax.axline((-100, -100), slope=1., color='red', ls='--')
    ax.set_aspect('equal')

    _ = plt.ylabel("Predicted MD [dB]")
    _ = plt.xlabel("True MD [dB]")


    limone = min(df_test.y.min(), df_test.y_pred.min()) - 1, max(df_test.y.max(), df_test.y_pred.max()) + 1
    _ = plt.xlim(limone)
    _ = plt.ylim(limone)

    if text is not None:
        plt.text(1.05, 0.5, text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    # plt.legend(loc=2, borderaxespad=0., handletextpad=0., fontsize='small', frameon=False)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, 'true_predictions_plot_only_test.png'))

    ############# FULL CV AND TEST ####################

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(data=df, x="y", y="y_pred", hue="dataset", ax=ax) #, style="n_slices", ax=ax)

    # ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
    ax.axline((-100, -100), slope=1., color='red', ls='--')
    ax.set_aspect('equal')

    _ = plt.ylabel("Predicted MD [dB]")
    _ = plt.xlabel("True MD [dB]")
    if lim is not None: 
        _ = plt.xlim(lim)
        _ = plt.ylim(lim)

    if text is not None:
        plt.text(1.05, 0.5, text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    plt.legend(loc=2, borderaxespad=0., handletextpad=0., fontsize='small', frameon=False)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, 'true_predictions_plot.png'))

def plot_mae_vs_glaucoma_stage(df, save_dir):
    fig, ax = plt.subplots()

    df['ae'] = np.abs(df.y - df.y_pred)
    sns.boxplot(x='stage', y='ae', hue='dataset', data=df, ax=ax, medianprops=dict(color="red"), palette='pastel')
    # ae = np.abs(df.y - df.y_pred)
    # df_box = df[df.dataset != 'test']
    # ae_box = ae[df.dataset != 'test']
    # sns.boxplot(x=df_box['stage'], y=ae_box, hue=df_box['dataset'], color='white', ax=ax, medianprops=dict(color="red"))
    # df_strip = df[df.dataset == 'test']
    # ae_strip = ae[df.dataset == 'test']
    # sns.stripplot(x=df_strip['stage'], y=ae_strip, hue=df_strip['dataset'], jitter=True, size=10, color='black', ax=ax)
    
    ax.set_ylabel('MAE [dB]')
    ax.set_xlabel('Glaucoma Stage')
    plt.legend(loc=9, fontsize="x-small") #bbox_to_anchor=(1.05, 1), borderaxespad=0.

    fig.savefig(os.path.join(save_dir, 'error_vs_stage_boxplot.png'), bbox_inches='tight')

def _plot_confusion_matrix(cm, classes, title, ax, cmap, normalize):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    # plt.colorbar(cax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes) #, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True GS')
    ax.set_xlabel('Predicted GS')

def plot_confusion_figure(cm, cm_test, classes, save_dir, text=None,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))    

    _plot_confusion_matrix(cm, classes, 'Validation', ax1, cmap, normalize)
    _plot_confusion_matrix(cm_test, classes, 'Test', ax2, cmap, normalize)

    if text is not None:
        plt.text(1.1, 0.5, text, horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

    