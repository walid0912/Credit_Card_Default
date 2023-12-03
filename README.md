# Data Preprocessing and Visualization

This repository contains code for loading and formatting data, as well as visualizing it in 2D using various techniques. The dataset used is the UCI Credit Card dataset.

## Data Loading and Formatting

### Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib

### Usage
1. Install the required dependencies using `pip install numpy pandas scikit-learn matplotlib`.
2. Download the UCI Credit Card dataset (`UCI_Credit_Card.csv`) from the source.
3. Place the dataset in the same directory as the code.

### Code Explanation

```python
import numpy as np
import pandas as pd

# Load the dataset
uci = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')

# Extract features and labels
X = uci.values[:, 1:-1]
Y = uci.values[:, -1]

# Shuffle the data
size = len(Y)
perm = np.arange(size)
np.random.shuffle(perm)

# Split the data into training, validation, and test sets
X_train = X[perm[0:20000], :]
Y_train = Y[perm[0:20000]]
X_val = X[perm[20000:25000], :]
Y_val = Y[perm[20000:25000]]
X_test = X[perm[25000:], :]
Y_test = Y[perm[25000:]]

# Display the shapes of the datasets
print(f'{X_train.shape} training samples, {X_val.shape} validation samples, {X_test.shape} test samples')
