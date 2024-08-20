import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Load dataset
df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Remove multicollinearity
def remove_multicollinearity(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif['features'] = range(X.shape[1])
    vif['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return X[:, vif[vif['VIF'] < 10]['features'].astype(int)]

X_reduced = remove_multicollinearity(X_scaled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# --- Perceptron - Classification (Binary) ---
def perceptron_train(X, y, lr=0.01, epochs=1000):
    weights = np.zeros(X.shape[1])
    bias = 0
    for _ in range(epochs):
        for i in range(X.shape[0]):
            prediction = np.dot(X[i], weights) + bias
            if (prediction >= 0) != y[i]:
                weights += lr * (y[i] - (prediction >= 0)) * X[i]
                bias += lr * (y[i] - (prediction >= 0))
    return weights, bias

def perceptron_predict(X, weights, bias):
    return np.dot(X, weights) + bias >= 0

weights_perceptron, bias_perceptron = perceptron_train(X_train, y_train)
predictions_perceptron = perceptron_predict(X_test, weights_perceptron, bias_perceptron)
accuracy_perceptron = accuracy_score(y_test, predictions_perceptron)
print(f'Perceptron Classification Accuracy: {accuracy_perceptron}')

# --- Sigmoid for Binary Classification ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_train(X, y, lr=0.01, epochs=1000):
    weights = np.zeros(X.shape[1])
    bias = 0
    for _ in range(epochs):
        linear_output = np.dot(X, weights) + bias
        predictions = sigmoid(linear_output)
        errors = y - predictions
        weights += lr * np.dot(X.T, errors)
        bias += lr * np.sum(errors)
    return weights, bias

def sigmoid_predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return sigmoid(linear_output) >= 0.5

weights_sigmoid, bias_sigmoid = sigmoid_train(X_train, y_train)
predictions_sigmoid = sigmoid_predict(X_test, weights_sigmoid, bias_sigmoid)
accuracy_sigmoid = accuracy_score(y_test, predictions_sigmoid)
print(f'Sigmoid Binary Classification Accuracy: {accuracy_sigmoid}')

# --- Multi-Class Classification with Sigmoid ---
def sigmoid_multi_class_train(X, y, lr=0.01, epochs=1000):
    num_classes = y.shape[1]
    weights = np.zeros((X.shape[1], num_classes))
    biases = np.zeros(num_classes)
    
    for _ in range(epochs):
        linear_output = np.dot(X, weights) + biases
        predictions = sigmoid(linear_output)
        errors = y - predictions
        weights += lr * np.dot(X.T, errors * predictions * (1 - predictions))
        biases += lr * np.sum(errors * predictions * (1 - predictions), axis=0)
    
    return weights, biases

def sigmoid_multi_class_predict(X, weights, biases):
    linear_output = np.dot(X, weights) + biases
    return sigmoid(linear_output)

# One-hot encode labels for multi-class
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y]

weights_sigmoid_multi, biases_sigmoid_multi = sigmoid_multi_class_train(X_train, y_one_hot)
predictions_sigmoid_multi = sigmoid_multi_class_predict(X_test, weights_sigmoid_multi, biases_sigmoid_multi)
predictions_sigmoid_multi_binary = (predictions_sigmoid_multi >= 0.5).astype(int)
accuracy_sigmoid_multi = np.mean(np.all(predictions_sigmoid_multi_binary == y_one_hot, axis=1))
print(f'Multi-Class Classification Accuracy with Sigmoid: {accuracy_sigmoid_multi}')

# --- Multi-Class Classification with Softmax ---
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

def softmax_train(X, y, lr=0.01, epochs=1000):
    num_features = X.shape[1]
    num_classes = y.shape[1]
    weights = np.zeros((num_features, num_classes))
    biases = np.zeros(num_classes)
    
    for _ in range(epochs):
        linear_output = np.dot(X, weights) + biases
        predictions = softmax(linear_output)
        loss = cross_entropy_loss(y, predictions)
        errors = predictions - y
        gradient_weights = np.dot(X.T, errors) / X.shape[0]
        gradient_biases = np.mean(errors, axis=0)
        weights -= lr * gradient_weights
        biases -= lr * gradient_biases
        
        if _ % 100 == 0:
            print(f'Epoch {_}: Loss = {loss}')
    
    return weights, biases

def softmax_predict(X, weights, biases):
    linear_output = np.dot(X, weights) + biases
    return softmax(linear_output)

weights_softmax, biases_softmax = softmax_train(X_train, y_one_hot)
predictions_softmax = softmax_predict(X_test, weights_softmax, biases_softmax)
predictions_softmax_class = np.argmax(predictions_softmax, axis=1)
y_test_class = np.argmax(y_one_hot, axis=1)
accuracy_softmax = accuracy_score(y_test_class, predictions_softmax_class)
print(f'Multi-Class Classification Accuracy with Softmax: {accuracy_softmax}')

# --- Ridge and Lasso Regression ---
def ridge_regression(X, y, alpha=1.0):
    I = np.identity(X.shape[1])
    weights = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return weights

def lasso_regression(X, y, alpha=1.0, iterations=1000, lr=0.01):
    weights = np.zeros(X.shape[1])
    for _ in range(iterations):
        predictions = np.dot(X, weights)
        errors = y - predictions
        gradient = -2 * np.dot(X.T, errors) / X.shape[0]
        weights -= lr * gradient
        weights = np.sign(weights) * np.maximum(0, np.abs(weights) - alpha * lr)
    return weights

# Ridge Regression
weights_ridge = ridge_regression(X_train, y_train)
predictions_ridge = np.dot(X_test, weights_ridge)
mse_ridge = mean_squared_error(y_test, predictions_ridge)
print(f'Ridge Regression Mean Squared Error: {mse_ridge}')

# Lasso Regression
weights_lasso = lasso_regression(X_train, y_train)
predictions_lasso = np.dot(X_test, weights_lasso)
mse_lasso = mean_squared_error(y_test, predictions_lasso)
print(f'Lasso Regression Mean Squared Error: {mse_lasso}')
