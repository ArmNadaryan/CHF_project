from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_model(X, y):
    """
    First:Splitting X and y to test and train
    Second:Creating and fitting the model
    Third: Predicting X_test
    :param X(independent columns):
    :param y(dependent column):
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # Found the best hyperparameters for RandomForestRegressor with GridSearch
    regressor = RandomForestRegressor(n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_depth=None)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test).reshape(-1, 1)
    return y_test, y_pred
