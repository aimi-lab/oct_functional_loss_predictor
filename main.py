from models import MDRegressor, MDRegressorClusters, MSRegressor, GSClassifier
from regression_enhanced_rf import RegressionEnhancedRandomForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import RFE
from constants import FEATURES, RNDM_STATE
import numpy as np

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
    
############################################
############################################

    # search_grid = {
    #     'max_features': [0.25], 
    #     'min_samples_leaf': [7], 
    #     'n_estimators': [100], 
    #     # 'criterion': ['squared_error', 'absolute_error']
    # }

    # for _ in range(50):
    #     regress = MDRegressor(
    #         'FEATURE_AUGMENTATION_REGRESSION_MD', 
    #         RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1), 
    #         feature_augmentation=True
    #         )
    #     regress.read_data(keep_age=False)
    #     regress.set_search_grid(**search_grid)
    #     regress.run()

############################################
############################################


    search_grid = {
        'max_features': [0.2, 0.3, 0.5, 0.7, 0.9, 1.0], 
        'min_samples_leaf': [1, 3, 5, 7, 9], 
        'n_estimators': [100, 300, 500], 
        # 'criterion': ['squared_error', 'absolute_error']
    }

    regress = MDRegressor(
        'RF_BEST_FEATURES', 
        RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1)
        )
    regress.read_data(
        keep_age=False, 
        features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD/00_important_features.txt'
        )
    regress.set_search_grid(**search_grid)
    regress.run('r2', random=True)

    regress = MDRegressor(
        'RF_VANILLA', 
        RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1)
        )
    regress.read_data(
        keep_age=False, 
        )
    regress.set_search_grid(**search_grid)
    regress.run('r2', random=True)

    regress = MDRegressor(
        'RF_W-AGE', 
        RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1)
        )
    regress.read_data(
        keep_age=True, 
        )
    regress.set_search_grid(**search_grid)
    regress.run('r2', random=True)

############################################
############################################

    # search_grid = {
    #     'max_depth': [4],
    #     'n_estimators': [250],
    #     'learning_rate': [0.05],
    #     "subsample":[0.75],
    #     "colsample_bytree":[0.75],
    #     "min_child_weight": [1], 
    #     "reg_lambda": [2]       
    # }

    # for _ in range(50):
    #     regress = MDRegressor(
    #         'FEATURE_AUGMENTATION_XGBOOST', 
    #         XGBRegressor(random_state=RNDM_STATE), 
    #         feature_augmentation=True
    #         )
    #     regress.read_data(keep_age=False)
    #     regress.set_search_grid(**search_grid)
    #     regress.run()

############################################
############################################

    # search_grid = {
    #     # 'max_depth': list(range(2, 10, 2)),
    #     # 'n_estimators': list(range(100, 260, 40)),
    #     'xgb__learning_rate': [0.1, 0.01, 0.05],
    #     # 'max_depth': [6],
    #     'xgb__n_estimators': [100],
    #     # 'learning_rate': [0.05],
    #     # "subsample":[1],
    #     # "colsample_bytree":[0.75],
    #     # "min_child_weight":[15],        
    #     # "subsample":[0.5, 0.75, 1],
    #     "xgb__colsample_bytree":[0.5, 0.75, 1],
    #     # "min_child_weight":[1, 5, 15],
    # }

    # scaler = StandardScaler()
    # xgb = XGBRegressor(random_state=RNDM_STATE)
    # xgbregr = Pipeline(steps=[('scaler', scaler), ('xgb', xgb)])

    # regress = MDRegressor(
    #     'XGBOOST_SCALED', 
    #     xgbregr
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     # features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD/00_important_features.txt'
    #     )
    # regress.set_search_grid(**search_grid)
    # regress.run()

    # search_grid = {
    #     'max_depth': list(range(2, 7, 1)),
    #     'n_estimators': list(range(50, 301, 50)),
    #     'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
    #     # 'max_depth': [6],
    #     # 'n_estimators': [100],
    #     # 'learning_rate': [0.05],
    #     # "subsample":[1],
    #     # "colsample_bytree":[0.75],
    #     # "min_child_weight":[15],        
    #     "subsample":[0.5, 0.75, 1],
    #     "colsample_bytree":[0.5, 0.75, 1],
    #     "reg_lambda": list(range(0, 21, 2)),
    #     # "gamma": [0.1, 0.3, 0.6],
    #     "min_child_weight":[0, 1, 2, 3],
    # }

    # regress = MDRegressor(
    #     'XGBOOST_ADIM', 
    #     XGBRegressor(random_state=RNDM_STATE)
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     # features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_XGBOOST/00_important_features.txt'
    #     )
    # regress.set_search_grid(**search_grid)
    # regress.run("r2", random=True)

    # regress = MDRegressor(
    #     'XGBOOST_BEST_FEAT', 
    #     XGBRegressor(random_state=RNDM_STATE)
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_XGBOOST/00_important_features.txt'
    #     )
    # regress.set_search_grid(**search_grid)
    # regress.run("r2", random=True)

    # regress = MDRegressor(
    #     'XGBOOST', 
    #     XGBRegressor(random_state=RNDM_STATE)
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     # features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD/00_important_features.txt'
    #     )
    # regress.set_search_grid(**search_grid)
    # regress.run()

    # regress = MDRegressor(
    #     'XGBOOST_NO-SEARCHGRID_BEST_FEATURES', 
    #     XGBRegressor(random_state=RNDM_STATE)
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD/00_important_features.txt'
    #     )
    # regress.set_search_grid(**search_grid)
    # regress.run()

    # regr_trans = TransformedTargetRegressor(
    #     # regressor=XGBRegressor(random_state=RNDM_STATE), func=np.log1p, inverse_func=np.expm1
    #     regressor=XGBRegressor(random_state=RNDM_STATE), transformer=StandardScaler()
    # )
    
    # rfe = RFE(estimator=DecisionTreeRegressor(random_state=RNDM_STATE), n_features_to_select=25, verbose=4)
    # search_grid = {'regressor__' + str(key): val for key, val in search_grid.items()}

    # regress = MDRegressor(
    #     'XGBOOST_RFE', 
    #     XGBRegressor(random_state=RNDM_STATE)
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     compute_weights=False
    #     # features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD/00_important_features.txt'
    #     )
    # regress.X_cv = rfe.fit_transform(regress.X_cv, regress.y_cv)
    # print(rfe.get_feature_names_out())
    # regress.X_test = rfe.transform(regress.X_test)
    # regress.set_search_grid(**search_grid)
    # regress.run('r2', random=True)

    # regress = MDRegressor(
    #     'XGBOOST_NO-SEARCHGRID', 
    #     XGBRegressor(random_state=RNDM_STATE)
    #     )
    # regress.read_data(
    #     keep_age=False,
    #     compute_weights=False
    #     # features_from='/home/davide/Dropbox (ARTORG)/CAS_Final_Project/src/outputs/FEATURE_AUGMENTATION_REGRESSION_MD/00_important_features.txt'
    #     )
    # regress.set_search_grid(**search_grid)
    # regress.run()

    # clust_regress = MDRegressorClusters('XGBOOST_CLUSTERS', XGBRegressor(random_state=RNDM_STATE))
    # clust_regress.read_data(keep_age=False)
    # clust_regress.set_search_grid(**search_grid)
    # clust_regress.run('r2')

############################################
############################################

    # search_grid = {
    #     'loss': ['linear', 'square', 'exponential'], 
    #     'learning_rate': [0.1, 1.0, 10.0], 
    #     'n_estimators': [50, 100, 200], 
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

    # search_grid = {
    #         'max_features': [0.7], 
    #         'min_samples_leaf': [3], 
    #         'n_estimators': [100], 
    #         # 'criterion': ['absolute_error']
    #         }

    # clust_regress = MDRegressorClusters('05_regression_clusters_rand_forest_wo_age_r2', RandomForestRegressor(random_state=RNDM_STATE, n_jobs=-1))
    # clust_regress.read_data(keep_age=False)
    # clust_regress.set_search_grid(**search_grid)
    # clust_regress.run('r2')

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

    # search_grid = {
    #     'max_features': [0.25], 
    #     'min_samples_leaf': [7], 
    #     # 'n_estimators': [100, 200, 300], 
    #     # 'criterion': ['squared_error', 'absolute_error']
    # }

    # regress = MDRegressor('03_regression_rerf_wo_age', RegressionEnhancedRandomForest(random_state=RNDM_STATE))
    # regress.read_data(keep_age=False)
    # regress.set_search_grid(**search_grid)
    # regress.run()

if __name__ == '__main__':
    main()