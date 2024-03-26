from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
def trainRandomForest(X_train,y_train,X_val,y_val):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Validar el modelo en el conjunto de validación
    y_pred_val = rf.predict(X_val)

    # Calcular métricas de rendimiento en el conjunto de validación
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"MSE (Validación): {mse}")
    print(f"R^2 (Validación): {r2}")

    return rf


