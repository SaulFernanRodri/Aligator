import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import pickle


def _create_pipeline(model, feature_selector=None):
    """Create a machine learning pipeline with optional feature selection."""
    steps = []
    if feature_selector:
        steps.append(('feature_selection', feature_selector))
    steps.append(('model', model))
    return Pipeline(steps)


def _perform_grid_search(pipeline, param_grid, x_train, y_train):
    """Perform grid search with cross-validation."""
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search


def _visualize_performance(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Performance of {model_name}')
    plt.legend()
    plt.show()


def _print_model_summary(model_name, mse, r2, target, filename, timestep_interval):
    with open(filename, 'a') as f:
        f.write(f"------- {timestep_interval} -------\n")
        f.write(f"--- Model Summary: {model_name} ---\n")
        f.write(f"Target: {target}\n")
        f.write(f"MSE (Validation): {mse}\n")
        f.write(f"R^2 (Validation): {r2}\n")
    print(f"------- {timestep_interval} -------")
    print(f"--- Model Summary: {model_name} ---")
    print(f"Target: {target}")
    print(f"MSE (Validation): {mse}")
    print(f"R^2 (Validation): {r2}")
    print("---------------------------------------\n")


def _print_best_params(search_cv, model_name):
    print(f"Best hiperparámeters {model_name}:", search_cv.best_params_)


def trainrandomforest(x_train, y_train, x_val, y_val, target, filename, timestep_interval):
    rf = RandomForestRegressor(random_state=42)
    pipeline = _create_pipeline(rf)
    param_dist_rf = {
        'model__n_estimators': [200, 300, 500, 1000],
        'model__max_depth': [20, 30, None],
        'model__min_samples_split': [2, 5, 10, 20],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt']
    }

    grid_search_rf = _perform_grid_search(pipeline, param_dist_rf, x_train, y_train)

    best_rf = grid_search_rf.best_estimator_

    _print_best_params(grid_search_rf, "Random Forest")

    y_pred_val = best_rf.predict(x_val)
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    _print_model_summary("Random Forest", mse, r2, target,
                         f"{filename}/model_RF_{target}.txt", timestep_interval)
    # _visualize_performance(y_val, y_pred_val, "Random Forest")

    return best_rf


def traingbr(x_train, y_train, x_val, y_val, target, filename, timestep_interval):
    gbr = GradientBoostingRegressor(random_state=42)
    pipeline = _create_pipeline(gbr)
    param_dist_gbr = {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [3, 5, 7, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4, 8]
    }

    grid_search_gbr = _perform_grid_search(pipeline, param_dist_gbr, x_train, y_train)
    best_gbr = grid_search_gbr.best_estimator_

    _print_best_params(grid_search_gbr, "Gradient Boosting Regressor")

    y_pred_val = best_gbr.predict(x_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    _print_model_summary("Gradient Boosting Regressor", mse_val, r2_val, target,
                         f"{filename}/model_GBR_{target}.txt", timestep_interval)
    # _visualize_performance(y_val, y_pred_val, "Gradient Boosting Regressor")

    return best_gbr


def trainsvr(x_train, y_train, x_val, y_val, target, filename, timestep_interval):
    svr = SVR()
    feature_selector = SelectFromModel(GradientBoostingRegressor(random_state=42))
    pipeline = _create_pipeline(svr, feature_selector)

    param_grid_svr = {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto'],
        'model__kernel': ['rbf', 'linear'],
        'model__epsilon': [0.01, 0.1, 1]
    }

    grid_search_svr = _perform_grid_search(pipeline, param_grid_svr, x_train, y_train)
    best_svr_model = grid_search_svr.best_estimator_

    _print_best_params(grid_search_svr, "Support Vector Regressor")

    y_pred_val = best_svr_model.predict(x_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    _print_model_summary("Support Vector Regressor", mse_val, r2_val, target,
                         f"{filename}/model_SVR_{target}.txt", timestep_interval)
    # _visualize_performance(y_val, y_pred_val, "Support Vector Regressor")
    return best_svr_model


def execute_model(x_train, y_train, x_val, y_val, target, results_folder, timesteps):
    rf = trainrandomforest(x_train, y_train, x_val, y_val, target, results_folder, timesteps)
    pickle.dump(rf, open(f"files/model/rf_{target}.pkl", 'wb'))

    gbr = traingbr(x_train, y_train, x_val, y_val, target, results_folder, timesteps)
    pickle.dump(gbr, open(f"files/model/gbr_{target}.pkl", 'wb'))

    svr = trainsvr(x_train, y_train, x_val, y_val, target, results_folder, timesteps)
    pickle.dump(svr, open(f"files/model/svr_{target}.pkl", 'wb'))
