import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Load dataset
df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels for classification
num_classes = len(np.unique(y))
y_one_hot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)

# --- MLP Classification ---
def create_mlp_classification(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),  # Hidden layer
        tf.keras.layers.Dense(32, activation='relu'),  # Another hidden layer
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer for classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train MLP model for classification
mlp_class = create_mlp_classification(X_train.shape[1], num_classes)
mlp_class.fit(X_train, y_train_class, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate MLP classification model
loss, accuracy = mlp_class.evaluate(X_test, y_test_class)
print(f'MLP Classification Accuracy: {accuracy}')

# --- MLP Regression ---
def create_mlp_regression(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),  # Hidden layer
        tf.keras.layers.Dense(32, activation='relu'),  # Another hidden layer
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create and train MLP model for regression
mlp_reg = create_mlp_regression(X_train.shape[1])
mlp_reg.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate MLP regression model
predictions_reg = mlp_reg.predict(X_test)
mse_reg = mean_squared_error(y_test, predictions_reg)
print(f'MLP Regression Mean Squared Error: {mse_reg}')
