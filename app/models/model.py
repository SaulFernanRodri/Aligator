from sklearn.model_selection import train_test_split

def model(df):
    X = df.drop(columns=['Timestep', 'Sector', 'Target CellSmutans', 'Target CellSmitis', 'Target Macromoleculeahl'])
    y = df['Target Macromoleculeahl']

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test
