import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# --- We only need the functions required for a forward pass ---
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


# --- 1. LOAD THE TRAINED WEIGHTS AND BIASES ---
print("--- Loading Saved Parameters... ---")
params = np.load("trained_params.npz")
W1 = params['W1']
b1 = params['b1']
W2 = params['W2']
b2 = params['b2']
print("--- Parameters Loaded Successfully ---")


# --- 2. LOAD THE TEST DATA ---
data = pd.read_csv('./train.csv')
data = np.array(data)
m, n = data.shape  # <-- THIS LINE WAS MISSING. We need to define 'n'.

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.


# --- 3. RUN THE TEST ---
def make_predictions(X, W1, b1, W2, b2):
    """Performs one forward pass to get predictions."""
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    """Tests a single example from the development set and shows the image."""
    current_image = X_dev[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_dev[index]
    print(f"Prediction: {prediction[0]}")
    print(f"Label: {label}")
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Get accuracy on the entire development set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Development Set Accuracy: {dev_accuracy}")

# Test a few individual examples
test_prediction(0, W1, b1, W2, b2)
test_prediction(42, W1, b1, W2, b2)
test_prediction(123, W1, b1, W2, b2)
