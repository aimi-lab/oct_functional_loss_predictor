from models import MDRegressor, MDRegressorClusters, MSRegressor, GSClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from constants import FEATURES, RNDM_STATE

def main():

#     search_grid = {
#             'max_features': [0.7, 0.8, 0.9, 1.0], 
#             'min_samples_leaf': [1, 3, 5, 7, 9], 
#             'n_estimators': [100, 150, 200], 
#             'criterion': ['squared_error', 'absolute_error']
#             }

#     regress = MSRegressor('00_regression_rand_forest_only_thicknesses_rnfl_gcl_MS', RandomForestRegressor(random_state=RNDM_STATE))
#     regress.read_data(features_in=[feat for feat in FEATURES if 'VOLUME' not in feat], layers_in=['RNFL', 'GCL+IPL'], keep_age=True)
#     regress.set_search_grid(**search_grid)
#     regress.run()

#     regress = MDRegressor('01_regression_rand_forest_only_thicknesses_rnfl_gcl', RandomForestRegressor(random_state=RNDM_STATE))
#     regress.read_data(features_in=[feat for feat in FEATURES if 'VOLUME' not in feat], layers_in=['RNFL', 'GCL+IPL'], keep_age=True)
#     regress.set_search_grid(**search_grid)
#     regress.run()

#     search_grid = {
#             'max_features': [0.2, 0.5, 0.7, 0.9, 1.0], 
#             'min_samples_leaf': [1, 3, 5, 7, 9], 
#             'n_estimators': [100, 300, 500], 
#             'criterion': ['squared_error', 'absolute_error']
#             }

    # search_grid = {
    #     'max_features': [0.2, 0.3, 0.4], 
    #     'min_samples_leaf': [3, 5, 7], 
    #     'n_estimators': [100, 200, 300], 
    #     # 'criterion': ['squared_error', 'absolute_error']
    # }

    # # regress = MDRegressor('02_regression_rand_forest_w_age', RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1))
    # # regress.read_data(keep_age=True)
    # # regress.set_search_grid(**search_grid)
    # # regress.run()

    # regress = MDRegressor('03_regression_rand_forest_wo_age', RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1))
    # regress.read_data(keep_age=False)
    # regress.set_search_grid(**search_grid)
    # regress.run()

    # search_grid = {
    #     # 'loss': ['linear', 'square', 'exponential'], 
    #     # 'learning_rate': [0.1, 1.0, 10.0], 
    #     # 'n_estimators': [50, 100, 200], 
    # }

    # regress = MDRegressor('ADABOOST', AdaBoostRegressor(random_state=RNDM_STATE))
    # regress.read_data(keep_age=False)
    # regress.set_search_grid(**search_grid)
    # regress.run()

    # scaler = StandardScaler()
    # svr = SVR()
    # svmregr = Pipeline(steps=[('scaler', scaler), ('svr', svr)])

    # search_grid = {
    #         'svr__C': [0.01, 0.1, 1.0, 10, 100], 
    #         'svr__kernel': ['linear'], 
    #         'svr__epsilon': [0.05, 0.1, 0.5, 1.0], 
    #         }

    # clust_regress = MDRegressorClusters('05_regression_clusters_svr_wo_age_r2', svmregr)
    # clust_regress.read_data(keep_age=False)
    # clust_regress.set_search_grid(**search_grid)
    # clust_regress.run('r2')

    # clust_regress = MDRegressorClusters('05_regression_clusters_svr_wo_age_mae', svmregr)
    # clust_regress.read_data(keep_age=False)
    # clust_regress.set_search_grid(**search_grid)
    # clust_regress.run()

    # search_grid = {
    #         'max_features': [0.2, 0.5, 0.7, 0.9, 1.0], 
    #         'min_samples_leaf': [1, 3, 5, 7, 9], 
    #         'n_estimators': [100, 300, 500], 
    #         'criterion': ['squared_error', 'absolute_error']
    #         }

    search_grid = {
            'max_features': [0.7], 
            'min_samples_leaf': [3], 
            'n_estimators': [100], 
            # 'criterion': ['absolute_error']
            }

    clust_regress = MDRegressorClusters('05_regression_clusters_rand_forest_wo_age_r2', RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1))
    clust_regress.read_data(keep_age=False)
    clust_regress.set_search_grid(**search_grid)
    clust_regress.run('r2')

    # clust_regress = MDRegressorClusters('05_regression_clusters_rand_forest_wo_age_mae', RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1))
    # clust_regress.read_data(keep_age=False)
    # clust_regress.set_search_grid(**search_grid)
    # clust_regress.run()

    # search_grid = {
    #         'max_features': [0.5, 0.7, 0.8], 
    #         'min_samples_leaf': [3, 5, 7, 20], 
    #         'n_estimators': [100, 200, 300], 
    #         # 'criterion': ['absolute_error']
    #         }

    # gs_classifier = GSClassifier(
    #     '10_classification_rand_forest_wo_age', 
    #     RandomForestClassifier(random_state=RNDM_STATE, class_weight="balanced", n_jobs=-1)
    #     )
    # gs_classifier.read_data(keep_age=False)
    # gs_classifier.set_search_grid(**search_grid)
    # gs_classifier.run()

if __name__ == '__main__':
    main()