
from sklearn.model_selection import train_test_split
def RandomForest(df):
    X = df.drop(columns=['Timestep', 'Sector', 'Target CellSmutans', 'Target CellSmitis', 'Target Macromoleculeahl'])
    y = df['Target CellSmutans']

    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test