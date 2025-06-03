# neural_network_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetworkClassifier(nn.Module):
    """
    A PyTorch-based neural network classifier with a scikit-learn-like API.
    
    This version builds a classifier with a final layer of 2 neurons (for "dead" and "alive")
    and uses CrossEntropyLoss for training. During inference, softmax is applied to convert logits
    to probability estimates.
    
    Parameters:
        input_dim (int): Number of input features.
        hidden_layers (list of int): List with the number of neurons in each hidden layer.
        dropout_rate (float): Dropout rate applied after each hidden layer.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device, optional): Device to run the model on.
    """
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001, device=None):
        super(NeuralNetworkClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Use provided device or default to GPU if available
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build network layers dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        # Output layer with 2 neurons for the two classes ("dead" and "alive")
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
        
        # Loss function: CrossEntropyLoss expects raw logits and class indices (0 or 1)
        # class_weights = torch.tensor([2.0, 3.0], device=self.device)  # e.g., w0=2.0, w1=1.0
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = None
        
        # Move the model to the appropriate device
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass: returns raw logits.
        """
        return self.network(x)
    
    def fit(self, X, y, epochs=10, batch_size=32, verbose=1):
        """
        Train the neural network on the provided data.
        
        Parameters:
            X (array-like or torch.Tensor): Input features of shape (n_samples, input_dim).
            y (array-like or torch.Tensor): Class labels as integers (0 or 1) of shape (n_samples,).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity flag.
        """
        # Convert X and y to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        # Ensure y is of shape (n_samples,)
        if y.dim() > 1:
            y = y.squeeze()
        
        # Create a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.train()  # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.forward(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_X.size(0)
            
            avg_loss = total_loss / len(dataloader.dataset)
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    def predict_proba(self, X):
        """
        Generate probability estimates for input samples using softmax.
        
        Parameters:
            X (array-like or torch.Tensor): Input features.
        
        Returns:
            np.ndarray: Array of shape (n_samples, 2) with probabilities for each class.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            logits = self.forward(X)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.cpu().numpy() 
    
    def predict(self, X):
        """
        Predict binary class labels for input samples.
        
        Parameters:
            X (array-like or torch.Tensor): Input features.
        
        Returns:
            np.ndarray: Array of predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
