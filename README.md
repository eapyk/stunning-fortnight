

## ❓ Question  
You are given a dataset with features \(X\) and target \(y\).  
Implement **Linear Regression from scratch using Gradient Descent** (without using `sklearn`).

### Requirements:
- Use Mean Squared Error (MSE) as the loss function  
- Update weights using gradient descent  
- Train for multiple iterations  
- Print the final weights  

---

## ✅ Answer (Python Implementation)

```python
import numpy as np

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Add bias term (column of 1s)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters
theta = np.zeros(X.shape[1])

# Hyperparameters
learning_rate = 0.01
iterations = 1000
m = len(y)

# Gradient Descent
for _ in range(iterations):
    predictions = X.dot(theta)
    errors = predictions - y
    
    gradients = (1/m) * X.T.dot(errors)
    theta -= learning_rate * gradients

print("Final Weights:", theta)
