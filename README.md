# Ai_logistic_regresstion_
Key Features: ✅ Binary classification with sigmoid function  ✅ Gradient descent optimization  ✅ L2 regularization support  ✅ Polynomial feature mapping for non-linear problems  ✅ Training visualization with cost curves
# Logistic Regression Implementation with Regularization

This repository contains a comprehensive implementation of logistic regression with regularization, built from scratch using NumPy. The code demonstrates both linear and non-linear classification problems with gradient descent optimization.

## Features

### Core Functions
- **Sigmoid Function**: Logistic function with numerical stability improvements (clipping to prevent overflow)
- **Cost Functions**: 
  - Binary cross-entropy loss (standard logistic regression)
  - Regularized version with L2 penalty
- **Gradient Calculations**:
  - Standard gradients for logistic regression
  - Regularized gradients for overfitting prevention
- **Gradient Descent**: Optimization algorithm with learning rate control
- **Polynomial Feature Mapping**: Creates polynomial features for non-linear decision boundaries
- **Prediction**: Binary classification with 0.5 threshold

### Implementation Highlights
1. **Handles Linear Classification**: For separable data with linear boundaries
2. **Handles Non-linear Classification**: Using polynomial feature expansion
3. **Regularization Support**: L2 regularization to prevent overfitting
4. **Comprehensive Visualization**: Training progress plots for both regular and regularized models
5. **Robust Error Handling**: Falls back to synthetic data if datasets are unavailable

## Key Components

### 1. **Sigmoid Function**
```python
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))
```

### 2. **Cost Functions**
- **Standard**: Binary cross-entropy loss
- **Regularized**: Adds L2 penalty term `(λ/2m) * Σw²`

### 3. **Feature Mapping**
Creates polynomial features up to specified degree (default: 6):
```python
def map_feature(X1, X2, degree=6):
    # Creates terms: X1^(i-j) * X2^j for i=1..degree, j=0..i
```

### 4. **Gradient Descent**
Iterative optimization with parameter updates:
- `w = w - α * ∂J/∂w`
- `b = b - α * ∂J/∂b`

## Usage

### Basic Training
```python
# Initialize parameters
w = np.zeros(X.shape[1])
b = 0

# Run gradient descent
w, b, J = gradient_descent(X, y, w, b, compute_cost, compute_gradient, 
                           alpha=0.001, iters=10000, lambda_=0)

# Make predictions
y_pred = predict(X, w, b)
accuracy = np.mean(y_pred == y) * 100
```

### With Regularization
```python
# Map features for non-linear problems
X_mapped = map_feature(X[:, 0], X[:, 1])

# Train with regularization
w2, b2, J2 = gradient_descent(X_mapped, y, w2, b2, compute_cost_reg, 
                              compute_gradient_reg, alpha=0.001, 
                              iters=5000, lambda_=1.0)
```

## Sample Output
```
Dataset 1 loaded: X shape (100, 2), y shape (100,)

Initial cost: 0.6931

Running gradient descent...
Iter 0 Cost 0.6930
Iter 500 Cost 0.6226
...
Iter 9500 Cost 0.2796

Training accuracy: 97.00%

Testing with regularization...
After feature mapping: X shape (100, 28)

Regularized cost (lambda=1.0): 0.6931

Running regularized gradient descent...
Iter 0 Cost 0.2470
...
Iter 4500 Cost 0.1323

Regularized model accuracy: 99.00%

All tests completed successfully!
```

## Visualization
The code generates two plots:
1. **Left**: Cost vs iterations for standard logistic regression
2. **Right**: Cost vs iterations for regularized logistic regression

## Dependencies
- NumPy (>= 1.19.0)
- Matplotlib (>= 3.3.0)

## File Structure
- `logistic_regression.py`: Main implementation file
- `data/`: Directory for dataset files (optional)
  - `ex2data1.txt`: Linear classification dataset
  - `ex2data2.txt`: Non-linear classification dataset

## Notes
- The code automatically creates synthetic data if dataset files are not found
- Regularization parameter `lambda` controls the strength of L2 penalty
- Learning rate `alpha` and iteration count `iters` can be tuned for better performance
- Polynomial degree in `map_feature` can be adjusted based on problem complexity

## Learning Concepts Demonstrated
1. **Binary Classification** using logistic regression
2. **Gradient Descent Optimization** for parameter learning
3. **Regularization** to prevent overfitting
4. **Feature Engineering** for non-linear problems
5. **Model Evaluation** through accuracy metrics
6. **Training Visualization** for monitoring convergence

This implementation serves as both a practical tool for binary classification and an educational resource for understanding logistic regression fundamentals.
