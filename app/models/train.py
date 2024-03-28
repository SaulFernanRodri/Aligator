from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
def trainRandomForest(X_train, y_train, X_val, y_val):
    rf = RandomForestRegressor(random_state=42)
    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    random_search_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=100, cv=3, random_state=42, n_jobs=-1)
    random_search_rf.fit(X_train, y_train)
    best_rf = random_search_rf.best_estimator_

    y_pred_val = best_rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"MSE (Validación) RF: {mse}")
    print(f"R^2 (Validación) RF: {r2}")

    return best_rf

def trainGBR(X_train, y_train, X_val, y_val):
    gbr = GradientBoostingRegressor(random_state=42)
    param_dist_gbr = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    random_search_gbr = RandomizedSearchCV(gbr, param_dist_gbr, n_iter=100, cv=3, random_state=42, n_jobs=-1)
    random_search_gbr.fit(X_train, y_train)
    best_gbr = random_search_gbr.best_estimator_

    y_pred_val = best_gbr.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    print(f"MSE (Validación) GBR: {mse_val}")
    print(f"R^2 (Validación) GBR: {r2_val}")

    return best_gbr


def trainSVR(X_train, y_train, X_val, y_val):
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(GradientBoostingRegressor(random_state=42))),
        ('svr', SVR())
    ])

    param_grid_svr = {
        'svr__C': [0.1, 1, 10],
        'svr__gamma': ['scale', 'auto'],
        'svr__kernel': ['rbf', 'linear'],
        'svr__epsilon': [0.01, 0.1, 1]
    }

    grid_search_svr = GridSearchCV(pipeline, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_svr.fit(X_train, y_train)

    print("Mejores hiperparámetros para SVR:", grid_search_svr.best_params_)

    best_svr_model = grid_search_svr.best_estimator_
    y_pred_val = best_svr_model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    print(f"MSE (Validación) SVR: {mse_val}")
    print(f"R^2 (Validación) SVR: {r2_val}")

    return best_svr_model
