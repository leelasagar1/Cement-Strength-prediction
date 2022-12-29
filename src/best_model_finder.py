from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import joblib
import os
import config


class Model_Finder:
    def __init__(self):
        pass

    #         linear_reg_model = LinearRegression()

    # Train Linear Regression
    def build_Linear_Regression_model(self, X_train, y_train):
        param_grid_linearReg = {"fit_intercept": [True, False], "copy_X": [True, False]}
        linear_reg_model = LinearRegression()
        grid = GridSearchCV(linear_reg_model, param_grid_linearReg, verbose=3, cv=5)
        grid.fit(X_train, y_train)
        fit_intercept = grid.best_params_["fit_intercept"]
        # normalize = grid.best_params_['normalize']
        copy_x = grid.best_params_["copy_X"]

        model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_x)
        model.fit(X_train, y_train)
        return model

    # train Random forest model
    def build_Random_Forest_model(self, X_train, y_train):
        param_grid_Random_forest_Tree = {
            "n_estimators": [10, 20, 30],
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_split": [2, 4, 8],
            "bootstrap": [True, False],
        }
        rf_reg = RandomForestRegressor()
        grid = GridSearchCV(rf_reg, param_grid_Random_forest_Tree, verbose=3, cv=5)
        grid.fit(X_train, y_train)
        n_estimators = grid.best_params_["n_estimators"]
        max_features = grid.best_params_["max_features"]
        min_samples_split = grid.best_params_["min_samples_split"]
        bootstrap = grid.best_params_["bootstrap"]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
        )
        model.fit(X_train, y_train)
        return model

    # train XGBoost model
    def build_XGBoost_model(self, X_train, y_train):
        param_grid = {
            "colsample_bytree": [0.3, 0.5, 0.8],
            "reg_alpha": [0, 0.5, 1, 5],
            "reg_lambda": [0, 0.5, 1, 5],
        }

        xgboost = XGBRegressor()
        grid = GridSearchCV(
            estimator=xgboost,
            param_grid=param_grid,
            scoring=["recall"],
            refit="recall",
            n_jobs=-1,
            cv=5,
            verbose=0,
        )

        grid_result = grid.fit(X_train, y_train)

        colsample_bytree = grid.best_params_["colsample_bytree"]
        reg_alpha = grid.best_params_["reg_alpha"]
        reg_lambda = grid.best_params_["reg_lambda"]

        model = XGBRegressor(
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

        model.fit(X_train, y_train)
        return model

    def find_best_model(self, data):
        X_train, X_test, y_train, y_test = data

        linear_model = self.build_Linear_Regression_model(X_train, y_train)
        linear_pred = linear_model.predict(X_test)
        linear_pred_error = r2_score(y_test, linear_pred)
        print("linear_pred_error:", linear_pred_error)

        rf_model = self.build_Random_Forest_model(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_pred_error = r2_score(y_test, rf_pred)
        print("rf_pred_error:", rf_pred_error)

        xg_model = self.build_XGBoost_model(X_train, y_train)
        xg_pred = xg_model.predict(X_test)
        xg_pred_error = r2_score(y_test, xg_pred)
        print("xg_pred_error:", xg_pred_error)

        models_error_dict = {}
        models_error_dict["linear_reg"] = linear_pred_error
        models_error_dict["rf"] = rf_pred_error
        models_error_dict["xg_boost"] = xg_pred_error

        models_dict = {}
        models_dict["linear_reg"] = linear_model
        models_dict["rf"] = rf_model
        models_dict["xg_boost"] = xg_model

        #     if rf_pred_error > linear_pred_error:
        #         model_name = f'Random_forest_{cluster}.pkl'
        #         joblib.dump(rf_model,os.path.join('models',model_name))
        #     else:
        #         model_name = f'Linear_Regression_{cluster}.pkl'
        #         joblib.dump(linear_model,os.path.join('models',model_name))
        model_name = max(models_error_dict, key=models_error_dict.get)

        return model_name, models_dict[model_name]

    def save_model(self, model_name, model, cluster):

        model_name = f"{model_name}_{cluster}.pkl"
        joblib.dump(model, os.path.join(config.model_save_location, model_name))
