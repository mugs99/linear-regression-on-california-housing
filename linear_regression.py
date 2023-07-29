import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_dataset():
    california = fetch_california_housing()
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['target'] = california.target
    return data

def perform_linear_regression(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

def plot_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True Values vs. Predictions")
    plt.show()

if __name__ == "__main__":
    dataset = load_dataset()

    target = dataset['target']
    features = dataset.drop('target', axis=1)

    print("Linear Regression Predictor")
    print("---------------------------")

    y_test, y_pred = perform_linear_regression(features, target)

    evaluate_model(y_test, y_pred)

    plot_results(y_test, y_pred)