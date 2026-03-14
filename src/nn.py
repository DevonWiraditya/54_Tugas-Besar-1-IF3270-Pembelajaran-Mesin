import random
from autodiff import Value
import functions as F

class Neuron:
    def __init__(self, nin, activation_name):
        self.w = F.uniform_init(nin, -1.0, 1.0)
        self.b = Value(random.uniform(-1.0, 1.0))
        self.activation_name = activation_name.lower()
        
        act_map = {
            "linear": F.linear,
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }
        self.activation = act_map.get(self.activation_name, F.linear)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act if self.activation_name == "softmax" else self.activation(act)

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, activation_name):
        self.activation_name = activation_name.lower()
        self.neurons = [Neuron(nin, self.activation_name) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        if self.activation_name == "softmax":
            return F.softmax(outs)
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class FFNN:
    def __init__(self, layer_sizes, activations_list):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations_list[i]) for i in range(len(layer_sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def fit(self, X_train, y_train, epochs, learning_rate, batch_size=1, loss_fn='mse'):
        loss_map = {
            'mse': F.mse,
            'bce': F.binary_crossentropy,
            'cce': F.categorical_crossentropy
        }
        criterion = loss_map.get(loss_fn.lower(), F.mse)
        history = {'loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                batch_loss = Value(0.0)
                for x, y in zip(X_batch, y_batch):
                    x_val = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
                    y_pred = self(x_val)
                    
                    y_list = y if isinstance(y, (list, tuple)) else [y]
                    y_pred_list = y_pred if isinstance(y_pred, (list, tuple)) else [y_pred]
                        
                    batch_loss += criterion(y_pred_list, y_list)
                
                batch_loss = batch_loss / Value(len(X_batch))
                
                for p in self.parameters():
                    p.grad = 0.0
                    
                batch_loss.backward()
                
                for p in self.parameters():
                    p.data -= learning_rate * p.grad
                    
                epoch_loss += batch_loss.data * len(X_batch)
                
            epoch_loss /= len(X_train)
            history['loss'].append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
            
        return history

    def predict(self, X):
        predictions = []
        for x in X:
            x_val = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
            pred = self(x_val)
            pred_list = pred if isinstance(pred, list) else [pred]
            predictions.append([p.data for p in pred_list])
        return predictions