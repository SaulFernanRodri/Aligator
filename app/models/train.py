from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt


def _visualize_performance(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Performance of {model_name}')
    plt.legend()
    plt.show()


def _print_model_summary(model_name, mse, r2, target, filename):
    with open(filename, 'a') as f:
        f.write(f"--- Model Summary: {model_name} ---\n")
        f.write(f"Target: {target}\n")
        f.write(f"MSE (Validation): {mse}\n")
        f.write(f"R^2 (Validation): {r2}\n")
    print(f"--- Model Summary: {model_name} ---")
    print(f"Target: {target}")
    print(f"MSE (Validation): {mse}")
    print(f"R^2 (Validation): {r2}")
    print("---------------------------------------\n")


def _print_best_params(search_cv, model_name):
    print(f"Best hiperparámeters {model_name}:", search_cv.best_params_)


def trainrandomforest(x_train, y_train, x_val, y_val, target):
    rf = RandomForestRegressor(random_state=42)
    param_dist_rf = {
        'n_estimators': [200, 300, 500, 1000],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt']
    }
    grid_search_rf = GridSearchCV(rf, param_dist_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_rf.fit(x_train, y_train)
    best_rf = grid_search_rf.best_estimator_

    _print_best_params(grid_search_rf, "Random Forest")

    y_pred_val = best_rf.predict(x_val)
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    _print_model_summary("Random Forest", mse, r2, target, "model_RF.txt")
    _visualize_performance(y_val, y_pred_val, "Random Forest")

    return best_rf


def traingbr(x_train, y_train, x_val, y_val, target):
    gbr = GradientBoostingRegressor(random_state=42)
    param_dist_gbr = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8]
    }
    grid_search_gbr = GridSearchCV(gbr, param_dist_gbr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_gbr.fit(x_train, y_train)
    best_gbr = grid_search_gbr.best_estimator_

    _print_best_params(grid_search_gbr, "Gradient Boosting Regressor")

    y_pred_val = best_gbr.predict(x_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    _print_model_summary("Gradient Boosting Regressor", mse_val, r2_val, target, "model_GBR.txt")
    _visualize_performance(y_val, y_pred_val, "Gradient Boosting Regressor")

    return best_gbr


def trainsvr(x_train, y_train, x_val, y_val, target):
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(GradientBoostingRegressor(random_state=42))),
        ('svr', SVR())
    ])

    param_grid_svr = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': ['scale', 'auto'],
        'svr__kernel': ['rbf', 'linear'],
        'svr__epsilon': [0.01, 0.1, 1]
    }

    grid_search_svr = GridSearchCV(pipeline, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_svr.fit(x_train, y_train)
    best_svr_model = grid_search_svr.best_estimator_

    _print_best_params(grid_search_svr, "Support Vector Regressor")

    y_pred_val = best_svr_model.predict(x_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    _print_model_summary("Support Vector Regressor", mse_val, r2_val, target, "model_SVR.txt")
    _visualize_performance(y_val, y_pred_val, "Support Vector Regressor")

    return best_svr_model
