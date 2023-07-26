"""
    implementation of MLP classifier
"""

import numpy as np


class MLPClassifierNP:
    def __init__(self, input_size, hidden_sizes, output_size, seed=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and biases for hidden layers
        self.hidden_weights = []
        self.hidden_biases = []
        prev_size = input_size
        for size in hidden_sizes:
            self.hidden_weights.append(np.random.randn(size, prev_size))
            self.hidden_biases.append(np.random.randn(size, 1))
            prev_size = size
        
        # Initialize weights and biases for output layer
        self.output_weights = np.random.randn(output_size, prev_size)
        self.output_bias = np.random.randn(output_size, 1)
    
    def forward(self, X):
        hidden_outputs = [X.T]
        for i in range(len(self.hidden_weights)):
            hidden_input = np.array(np.dot(self.hidden_weights[i], hidden_outputs[-1]) + self.hidden_biases[i], float)
            hidden_output = 1 / (1 + np.exp(-hidden_input))
            hidden_outputs.append(hidden_output)

        output = np.array(np.dot(self.output_weights, hidden_outputs[-1]) + self.output_bias, float)
        output = 1 / (1 + np.exp(-output))
        return output, hidden_outputs
    
    
    def fit(self, X, y, learning_rate=0.01, epochs=100, verbose = False):
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(epochs):
            # Forward pass
            output, hidden_outputs = self.forward(X)
            
            # Calculate loss and accuracy
            loss = -np.mean(y.T * np.log(output) + (1 - y.T) * np.log(1 - output))
            epoch_losses.append(loss)
            accuracy = np.mean((output >= 0.5) == y.T)
            epoch_accuracies.append(accuracy)
        
            # Backward pass through the output layer
            output_error = (y.T - output) * output * (1 - output)
            output_delta = learning_rate * np.dot(output_error, hidden_outputs[-1].T)
            self.output_weights += output_delta
            self.output_bias += learning_rate * np.sum(output_error, axis=1, keepdims=True)
            
            # Backward pass through the hidden layers
            hidden_error = np.dot(self.output_weights.T, output_error) * hidden_outputs[-1] * (1 - hidden_outputs[-1])
            for i in range(2, len(self.hidden_weights)+1):
                hidden_delta = learning_rate * np.dot(hidden_error, hidden_outputs[-i].T)
                self.hidden_weights[-i+1] += hidden_delta
                self.hidden_biases[-i+1] += learning_rate * np.sum(hidden_error, axis=1, keepdims=True)
                hidden_error = np.dot(self.hidden_weights[-i+1].T, hidden_error) * hidden_outputs[-i] * (1 - hidden_outputs[-i])
            
        return epoch_losses, epoch_accuracies
    
    def k_fold_cross_val_score(self, X, y, k=5, learning_rate=0.01, epochs=100, seed=None):
        indices = np.arange(X.shape[0])
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        fold_size = X.shape[0] // k
        scores = []
        for i in range(k):
            fold_start = i * fold_size
            if i == k-1:
                fold_end = None
            else:
                fold_end = (i+1) * fold_size
            fold_indices = indices[fold_start:fold_end]
            X_train = np.delete(X, fold_indices, axis=0)
            y_train = np.delete(y, fold_indices, axis=0)
            X_val = X[fold_indices]
            y_val = y[fold_indices]
            clf = MLPClassifierNP(self.input_size, self.hidden_sizes, self.output_size)
            clf.fit(X_train, y_train, learning_rate=learning_rate, epochs=epochs)
            y_pred = clf.predict(X_val)
            acc = np.mean(y_pred == y_val)
            scores.append(round(acc, 4))
        return round(np.mean(scores), 4), scores

    
    def predict(self, X):
        output = self.forward(X)[0]
        return np.round(output).T
    
    
