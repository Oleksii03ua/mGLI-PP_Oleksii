import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor 


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GBDT(torch.nn.Module):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.estimators = []
            self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        residual = y - self.initial_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            self.estimators.append(tree)
            residual -= self.learning_rate * update

    def predict(self, X):
        if self.initial_prediction is None:
            raise ValueError("Model is not fitted yet.")

        y_pred = self.initial_prediction
        for tree in self.estimators:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred