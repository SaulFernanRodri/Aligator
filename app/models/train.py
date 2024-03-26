from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
def trainRandomForest(X_train,y_train,X_val,y_val):

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_val = rf.predict(X_val)

    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"MSE (Validación): {mse}")
    print(f"R^2 (Validación): {r2}")

    return rf
def trainGBR(X_train,y_train,X_val,y_val):
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)

    y_pred_val = gbr.predict(X_val)

    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    print(f"MSE (Validación): {mse_val}")
    print(f"R^2 (Validación): {r2_val}")

    return gbr
def trainSVR(X_train,y_train,X_val,y_val):
    param_grid_svr = {
        'C': [1, 10, 100, 1000],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    svr = SVR()

    grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_squared_error')
    grid_search_svr.fit(X_train, y_train)

    print("Best hyperparameters for SVR:", grid_search_svr.best_params_)

    best_svr_model = grid_search_svr.best_estimator_

    y_pred_val = best_svr_model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    print(f"MSE (Validation): {mse_val}")
    print(f"R^2 (Validation): {r2_val}")
    return best_svr_model




