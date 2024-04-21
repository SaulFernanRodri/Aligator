from sklearn.model_selection import train_test_split


def modeling(df):
    targets = [col for col in df.columns if col.startswith('Target')]
    results = {}

    for target in targets:
        x = df.drop(columns=['Timestep', 'Sector', target])
        y = df[target]

        x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # 0.25 * 0.8 = 0.2
        x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.25, random_state=42)

        results[target] = {
            'x_train': x_train,
            'x_val': x_val,
            'x_test': x_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    return targets, results
