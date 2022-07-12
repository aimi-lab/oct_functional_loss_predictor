from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

class RegressionEnhancedRandomForest(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1.0, n_estimators=100, min_samples_leaf=1, max_features='auto', criterion='absolute_error', random_state=None): 
        # had to introduce random_state to pass tests
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion

    def fit(self, X, y):

        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[-1]

        self.lasso_ = Lasso(alpha=self.alpha, random_state=self.random_state)
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion, 
            max_features=self.max_features, 
            min_samples_leaf=self.min_samples_leaf, 
            random_state=self.random_state,
            n_jobs=-1
            )

        self.lasso_.fit(X, y)
        eps = y - self.lasso_.predict(X)
        self.rf_.fit(X, eps)

        return self

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        return self.lasso_.predict(X) + self.rf_.predict(X)

if __name__ == '__main__':
    check_estimator(RegressionEnhancedRandomForest(random_state=42))