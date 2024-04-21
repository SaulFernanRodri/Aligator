from sklearn.metrics import mean_squared_error, r2_score


def test(model, x_test, y_test):
    y_pred_test = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f"MSE (Test): {mse}")
    print(f"R^2 (Test): {r2}")
    print("---------------------------------------\n")
