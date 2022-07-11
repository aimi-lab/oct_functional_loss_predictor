from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score, cross_validate, StratifiedGroupKFold, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer, cohen_kappa_score, confusion_matrix, accuracy_score, balanced_accuracy_score
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import misc
import constants as myc
import plotting_utils
from regression_enhanced_rf import RegressionEnhancedRandomForest

sns.set_context('poster')


class AbstractModel(ABC):

    def __init__(self, save_dir, model) -> None:
        self.save_dir = misc.create_dir_if_not_exist(save_dir)
        self.model = model

        self.df_cv = None
        self.X_cv = None
        self.y_cv = None

        self.df_test = None
        self.X_test = None
        self.y_test = None

        self.gkf_cv = None
        self.search_grid = None

        self.best_model = None

        self.y_cv_pred = None
        self.y_test_pred = None
        self.scores = None
        self.metrics = None

    def read_data(self, keep_age, target):
        self.df_cv, self.df_test = misc.read_dataset()
        self.gkf_cv = list(StratifiedGroupKFold().split(
            self.df_cv, 
            self.df_cv['GS'], 
            groups=self.df_cv.index.get_level_values(0)
            ))

        regex = 'THICKNESS|VOLUME'
        if keep_age: regex += '|Age'

        self.X_cv = self.df_cv.filter(regex=regex)
        self.X_test = self.df_test.filter(regex=regex)
        self.y_cv = self.df_cv[target]
        self.y_test = self.df_test[target]

        df_full = pd.concat([self.df_test, self.df_cv])
        print(f"Unique patients identified: {len(df_full.index.get_level_values(0).unique())}")
        print(f"Number of samples: {len(df_full)}")
        print(f"Number of features: {len(self.X_cv.columns)}")

        fig, ax = plt.subplots(1, 3, figsize=[12, 8])
        sns.histplot(df_full["GS"], bins=[0, 1, 2, 3, 4, 5, 6], ax=ax[0])
        sns.histplot(self.df_cv["GS"], bins=[0, 1, 2, 3, 4, 5, 6], ax=ax[1])
        sns.histplot(self.df_test["GS"], bins=[0, 1, 2, 3, 4, 5, 6], ax=ax[2])
        for aaxx in ax:
            aaxx.set_xticks([0, 1, 2, 3, 4, 5])
            aaxx.set_xlabel('Glaucoma Stage')
        ax[0].set_title('Full Dataset')
        ax[1].set_title('CV Dataset')
        ax[2].set_title('Test Dataset')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'dataset_histplot.png'))

    def set_search_grid(self, **kwargs):
        self.search_grid = kwargs

    def run_grid_search(self, scoring):
        return misc.run_grid_search(self.X_cv, self.y_cv, self.model, self.gkf_cv, self.search_grid, scoring)

    @abstractmethod
    def run(self, scoring='neg_mean_absolute_error'):
        if self.search_grid is None: 
            raise AttributeError('Set grid search first!')

        self.best_model = self.run_grid_search(scoring)
        self.y_cv_pred = cross_val_predict(self.best_model, self.X_cv, self.y_cv, cv=self.gkf_cv)
        self.scores = cross_validate(self.best_model, self.X_cv, self.y_cv, cv=self.gkf_cv, scoring=self.metrics)

        self.y_test_pred = self.best_model.predict(self.X_test)


class Regressor(AbstractModel):

    def __init__(self, save_dir, model) -> None:
        super().__init__(save_dir, model)
        self.metrics = ['neg_mean_absolute_error', 'r2']

    def run(self, scoring='neg_mean_absolute_error'):
        super().run(scoring)
        mae_test = mean_absolute_error(self.y_test, self.y_test_pred)
        r2_test = r2_score(self.y_test, self.y_test_pred)
        
        text = ''
        best_params = self.best_model.get_params()
        for k in sorted(self.search_grid.keys()):
            v = best_params[k]
            if isinstance(v, str):
                vv = myc.ERROR_ABBR.get(v, v)
                text += f'{k}: {vv}\n'
            elif isinstance(v, int):
                text += f'{k}: {v:d}\n'
            elif isinstance(v, float):
                text += f'{k}: {v:.2f}\n'
        text += '\n'
        text += f'MAE$_{{CV}}$: {-self.scores["test_neg_mean_absolute_error"].mean():.2f} $\pm$ {self.scores["test_neg_mean_absolute_error"].std():.2f}\n'
        text += f'MAE$_{{test}}$: {mae_test:.2f}\n'
        text += f'$R^2_{{CV}}$: {(self.scores["test_r2"].mean()):.2f} $\pm$ {(self.scores["test_r2"].std()):.2f}\n'
        text += f'$R^2_{{test}}$: {r2_test:.2f}'


        df_cv = pd.DataFrame({
            'y': self.y_cv, 
            'y_pred': self.y_cv_pred, 
            'dataset': f'{myc.CV}-fold CV', 
            # 'n_slices': self.df_cv['slices'], 
            'stage': self.df_cv['GS']})
        df_test = pd.DataFrame({
            'y': self.y_test, 
            'y_pred': self.y_test_pred, 
            'dataset': 'test', 
            # 'n_slices': self.df_test['slices'], 
            'stage': self.df_test['GS']})
        df_full = pd.concat([df_cv, df_test])

        plotting_utils.plot_mae_vs_glaucoma_stage(df_full, self.save_dir)
        plotting_utils.plot_truth_prediction(
            df_full, self.save_dir, text=text, 
            lim=[
                min(self.y_cv.min(), self.y_cv_pred.min(), self.y_test.min(), self.y_test_pred.min()) - 1, 
                max(self.y_cv.max(), self.y_cv_pred.max(), self.y_test.max(), self.y_test_pred.max()) + 1
                ])
    
        if not isinstance(self.model, RegressionEnhancedRandomForest): 
            plotting_utils.plot_feature_importance(self.X_cv, self.y_cv, self.best_model, myc.CV, self.save_dir)


class MDRegressor(Regressor):
    
    def read_data(self, keep_age=False):
        super().read_data(keep_age, 'MD')


class MSRegressor(Regressor):
    
    def read_data(self, keep_age=False):
        super().read_data(keep_age, 'MS')


class MDRegressorClusters(Regressor):

    def __init__(self, save_dir, model) -> None:
        super().__init__(save_dir, model)
        self.metrics = ['neg_mean_absolute_error', 'r2']
        self.clusters = range(1, 11)

    def read_data(self, keep_age=False):
        super().read_data(keep_age, 'MD')
        del self.y_cv, self.y_test

    def run(self, scoring='neg_mean_absolute_error'):
        
        self.results_table = pd.DataFrame({}, columns=['cluster', 'metric', 'dataset', 'metric_value'])

        clust_base_dir = self.save_dir

        for cluster in self.clusters:
            
            self.save_dir = misc.create_dir_if_not_exist(os.path.join(clust_base_dir, f'cluster_{cluster:02d}'), prefix=False)

            self.y_cv = self.df_cv[f'Cluster {cluster}']
            self.y_test = self.df_test[f'Cluster {cluster}']

            super().run(scoring)

            # mape_cv = mean_absolute_percentage_error(self.y_cv, self.y_cv_pred, multioutput='raw_values')
            # mape_test = mean_absolute_percentage_error(self.y_test, self.y_test_pred, multioutput='raw_values')
            mae_cv = abs(self.y_cv - self.y_cv_pred)
            mae_test = abs(self.y_test - self.y_test_pred)

            df_cv = pd.DataFrame({'cluster': cluster, 'metric': 'MD', 'dataset': 'CV', 'metric_value': self.y_cv, 'GS': self.df_cv['GS']})
            df_te = pd.DataFrame({'cluster': cluster, 'metric': 'MD', 'dataset': 'Test', 'metric_value': self.y_test, 'GS': self.df_test['GS']})
            df_mae_cv = pd.DataFrame({'cluster': cluster, 'metric': 'MAE', 'dataset': 'CV', 'metric_value': mae_cv, 'GS': self.df_cv['GS']})
            df_mae_te = pd.DataFrame({'cluster': cluster, 'metric': 'MAE', 'dataset': 'Test', 'metric_value': mae_test, 'GS': self.df_test['GS']})

            self.results_table = pd.concat([self.results_table, df_cv, df_te, df_mae_cv, df_mae_te], ignore_index=True)

        self.results_table.to_csv(os.path.join(clust_base_dir, 'results.csv'))
        print('ciao')


class GSClassifier(AbstractModel):

    def __init__(self, save_dir, model) -> None:
        super().__init__(save_dir, model)

        qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
        self.metrics = {'kappa': qwk_scorer, 'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy'}

    def read_data(self, keep_age=False, smote=False):

        # For classification, y is glaucoma stage
        # returned features_cv, target_cv, glauc_stage_cv, extras_cv, features_test, target_test, glauc_stage_test, extras_test
        super().read_data()

        if smote:
            assert self.model.rf_weights is None, 'Cannot apply SMOTE with RF weights'
            sm = SMOTE(random_state=myc.RNDM_STATE, k_neighbors=5)
            self.X_cv, self.y_cv = sm.fit_resample(self.X_cv, self.y_cv)
            # how to stratify synythetic data on patient?
            self.gkf_cv = list(StratifiedKFold(n_splits=myc.CV).split(self.X_cv, self.y_cv))
        else:
            self.gkf_cv = list(StratifiedGroupKFold(n_splits=myc.CV).split(self.X_cv, self.y_cv, groups=self.df_cv.index.get_level_values(0)))

    def run(self):
        super().run(scoring=self.metrics['kappa'])

        cnf_matrix_cv = confusion_matrix(self.y_cv, self.y_cv_pred, labels=range(6))
        cnf_matrix_test = confusion_matrix(self.y_test, self.y_test_pred, labels=range(6))
        
        text = ''
        best_params = self.best_model.get_params()
        for k in sorted(self.search_grid.keys()):
            v = best_params[k]
            if isinstance(v, str):
                vv = myc.ERROR_ABBR.get(v, v)
                text += f'{k}: {vv}\n'
            elif isinstance(v, int):
                text += f'{k}: {v:d}\n'
            elif isinstance(v, float):
                text += f'{k}: {v:.2f}\n'
        text += '\n'
        # text += f'ACC$_{{CV}}$: {self.scores["test_accuracy"].mean():.2f} $\pm$ {self.scores["test_accuracy"].std():.2f}\n'
        # text += f'ACC$_{{test}}$: {accuracy_score(self.y_test, self.y_test_pred):.2f}\n'
        # text += f'MAE$_{{test}}$: {mae_test:.2f}\n'
        # text += f'$R^2_{{CV}}$: {(self.scores["test_r2"].mean() * 100):.2f} $\pm$ {(self.scores["test_r2"].std() * 100):.2f}'
        
        print(self.scores.keys())

        text = u"ACC*$_{{CV}}$: %0.2f $\pm$ %0.2f\nACC*$_{{test}}$: %0.2f\nQWK$_{{CV}}$: %0.2f $\pm$ %0.2f\nQWK$_{{test}}$: %0.2f" % (
            self.scores['test_balanced_accuracy'].mean(), 
            self.scores['test_balanced_accuracy'].std(),
            balanced_accuracy_score(self.y_test, self.y_test_pred), 
            self.scores['test_kappa'].mean(), 
            self.scores['test_kappa'].std(),
            cohen_kappa_score(self.y_test.astype(int).values, self.y_test_pred, weights="quadratic")
        )

        plotting_utils.plot_confusion_figure(cnf_matrix_cv, cnf_matrix_test, range(6), self.save_dir, text)

        # df_cv = pd.DataFrame({'y': y_cv, 'y_pred': y_pred, 'dataset': f'{CV}-fold CV', 'n_slices': extras_cv['n_slices']})
        # df_test = pd.DataFrame({'y': y_test, 'y_pred': y_pred_test, 'dataset': f'test', 'n_slices': extras_test['n_slices']})
        # df_full = pd.concat([df_cv, df_test])

        # fig = plotting_utils.plot_truth_prediction(
        #     df_full, text=text, 
        #     lim=[min(y_pred.min(), y_cv.min(), y_pred_test.min(), y_test.min()) - 1, max(y_pred.max(), y_cv.max(), y_pred_test.max(), y_test.max()) + 1])
        # fig.savefig(os.path.join(save_dir, 'true_predictions_plot.png'))

        plotting_utils.plot_feature_importance(self.X_cv, self.y_cv, self.best_model, myc.CV, self.save_dir)