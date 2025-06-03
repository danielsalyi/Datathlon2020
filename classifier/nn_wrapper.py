import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from classifier.nn_classifier import NeuralNetworkClassifier 


class NNClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn wrapper for a PyTorch NeuralNetworkClassifier.
    Exposes hyperparameters such as hidden_layers, dropout_rate, epochs, etc.
    """
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.2,
                 learning_rate=0.001, epochs=10, batch_size=32, device=None, verbose=0):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None

    def fit(self, X, y):
        if self.verbose:
            print("Starting training with hyperparameters:")
            print(f"  hidden_layers: {self.hidden_layers}")
            print(f"  dropout_rate: {self.dropout_rate}")
            print(f"  learning_rate: {self.learning_rate}")
            print(f"  epochs: {self.epochs}")
            print(f"  batch_size: {self.batch_size}")
        
        # Build a new model with the current hyperparameters
        self.model_ = NeuralNetworkClassifier(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            device=self.device
        )
        # Train the model
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        # Return predictions from the PyTorch model
        return self.model_.predict(X)

    def predict_proba(self, X):
        # Return probability estimates from the model
        return self.model_.predict_proba(X)

    def score(self, X, y):
        # Default scoring: accuracy
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)