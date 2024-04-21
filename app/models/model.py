from sklearn.model_selection import train_test_split


def model(df):
    x = df.drop(columns=['Timestep', 'Sector', 'Target CellSmutans', 'Target CellSmitis', 'Target Macromoleculeahl'])
    y = df['Target CellSmutans']

    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 0.25 * 0.8 = 0.2
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.25, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test
