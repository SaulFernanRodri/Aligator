from sklearn.metrics import mean_squared_error, r2_score
def testRandomForest(model, X_test, y_test):
    # Predecir en el conjunto de prueba
    y_pred_test = model.predict(X_test)

    # Calcular métricas de regresión
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f"MSE (Test): {mse}")
    print(f"R^2 (Test): {r2}")

    # Devolver las predicciones y las métricas
    return y_pred_test, mse, r2