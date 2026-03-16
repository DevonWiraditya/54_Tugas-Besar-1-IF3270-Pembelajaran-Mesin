import random
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from autodiff import Value
import functions as F

class Neuron:
    def __init__(self, nin, activation_name, init_method='uniform', init_params=None, seed=None):
        """
        Parameters:
        - nin: number of inputs
        - activation_name: 'linear', 'relu', 'sigmoid', 'tanh', 'softmax'
        - init_method: 'zero', 'uniform', 'normal'
        - init_params: dict with init-specific params
            - uniform: {'low': float, 'high': float}
            - normal: {'mean': float, 'variance': float}
        - seed: random seed for reproducibility
        """
        self.nin = nin
        self.activation_name = activation_name.lower()

        if init_params is None:
            init_params = {}

        if init_method == 'zero':
            self.w = F.zero_init(nin)
            self.b = Value(0.0)
        elif init_method == 'uniform':
            low = init_params.get('low', -1.0)
            high = init_params.get('high', 1.0)
            self.w = F.uniform_init(nin, low, high, seed=seed)
            self.b = Value(random.uniform(low, high))
        elif init_method == 'normal':
            mean = init_params.get('mean', 0.0)
            variance = init_params.get('variance', 1.0)
            self.w = F.normal_init(nin, mean, variance, seed=seed)
            self.b = Value(random.gauss(mean, variance**0.5))
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

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

    def weights(self):
        return self.w

    def bias(self):
        return self.b


class Layer:
    def __init__(self, nin, nout, activation_name, init_method='uniform', init_params=None, seed=None):
        self.nin = nin
        self.nout = nout
        self.activation_name = activation_name.lower()
        self.neurons = [Neuron(nin, self.activation_name, init_method, init_params, seed) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        if self.activation_name == "softmax":
            return F.softmax(outs)
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def weights(self):
        return [w for neuron in self.neurons for w in neuron.weights()]

    def biases(self):
        return [neuron.bias() for neuron in self.neurons]


class FFNN:
    def __init__(self, layer_sizes, activations_list, init_method='uniform', init_params=None, seed=None):
        """
        Parameters:
        - layer_sizes: list of int, e.g. [784, 128, 64, 10]
        - activations_list: list of str, one per layer (excluding input), e.g. ['relu', 'relu', 'softmax']
        - init_method: 'zero', 'uniform', 'normal'
        - init_params: dict for init method, e.g. {'low': -0.5, 'high': 0.5}
        - seed: random seed
        """
        assert len(activations_list) == len(layer_sizes) - 1, \
            f"Need {len(layer_sizes)-1} activations, got {len(activations_list)}"

        self.layer_sizes = layer_sizes
        self.activations_list = activations_list
        self.init_method = init_method
        self.init_params = init_params

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations_list[i],
                         init_method, init_params, seed)
            self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def _compute_reg_loss(self, reg_type, reg_lambda):
        """Compute regularization penalty."""
        if reg_type is None or reg_lambda == 0:
            return Value(0.0)

        reg_loss = Value(0.0)
        if reg_type == 'l1':
            for p in self.parameters():
                reg_loss = reg_loss + Value(abs(p.data))
        elif reg_type == 'l2':
            for p in self.parameters():
                reg_loss = reg_loss + p ** 2
        else:
            raise ValueError(f"Unknown reg_type: {reg_type}. Use 'l1', 'l2', or None.")

        return reg_loss * reg_lambda

    def fit(self, X_train, y_train, epochs, learning_rate, batch_size=1,
            loss_fn='mse', X_val=None, y_val=None, verbose=1,
            reg_type=None, reg_lambda=0.0):
        """
        Train the model.

        Parameters:
        - X_train, y_train: training data
        - epochs: number of epochs
        - learning_rate: learning rate for gradient descent
        - batch_size: mini-batch size
        - loss_fn: 'mse', 'bce', 'cce'
        - X_val, y_val: optional validation data
        - verbose: 0 = silent, 1 = progress bar with loss
        - reg_type: None, 'l1', 'l2'
        - reg_lambda: regularization strength

        Returns:
        - history: {'train_loss': [...], 'val_loss': [...]}
        """
        loss_map = {
            'mse': F.mse,
            'bce': F.binary_crossentropy,
            'cce': F.categorical_crossentropy
        }
        criterion = loss_map.get(loss_fn.lower(), F.mse)
        history = {'train_loss': [], 'val_loss': []}

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = len(X_train)

            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            X_shuffled = [X_train[i] for i in indices]
            y_shuffled = [y_train[i] for i in indices]

            n_batches = (n_samples + batch_size - 1) // batch_size

            if verbose == 1 and has_tqdm:
                batch_iter = tqdm(range(0, n_samples, batch_size),
                                 desc=f"Epoch {epoch+1}/{epochs}",
                                 total=n_batches, leave=True)
            else:
                batch_iter = range(0, n_samples, batch_size)

            for i in batch_iter:
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                batch_loss = Value(0.0)
                for x, y in zip(X_batch, y_batch):
                    x_val = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
                    y_pred = self(x_val)

                    y_list = y if isinstance(y, (list, tuple)) else [y]
                    y_pred_list = y_pred if isinstance(y_pred, (list, tuple)) else [y_pred]

                    batch_loss = batch_loss + criterion(y_pred_list, y_list)

                batch_loss = batch_loss / Value(len(X_batch))

                # Add regularization
                reg_loss = self._compute_reg_loss(reg_type, reg_lambda)
                total_loss = batch_loss + reg_loss

                # Zero gradients
                for p in self.parameters():
                    p.grad = 0.0

                # Backward pass
                total_loss.backward()

                # Update weights
                for p in self.parameters():
                    p.data -= learning_rate * p.grad

                epoch_loss += batch_loss.data * len(X_batch)

                if verbose == 1 and has_tqdm and isinstance(batch_iter, tqdm):
                    batch_iter.set_postfix({'loss': f'{batch_loss.data:.4f}'})

            # Average training loss for this epoch
            avg_train_loss = epoch_loss / n_samples
            history['train_loss'].append(avg_train_loss)

            # Compute validation loss
            if X_val is not None and y_val is not None:
                val_loss = 0.0
                for x, y in zip(X_val, y_val):
                    x_v = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
                    y_pred = self(x_v)
                    y_list = y if isinstance(y, (list, tuple)) else [y]
                    y_pred_list = y_pred if isinstance(y_pred, (list, tuple)) else [y_pred]
                    val_loss += criterion(y_pred_list, y_list).data
                avg_val_loss = val_loss / len(X_val)
                history['val_loss'].append(avg_val_loss)
            else:
                history['val_loss'].append(None)

            if verbose == 1:
                val_str = f" | Val Loss: {history['val_loss'][-1]:.4f}" if history['val_loss'][-1] is not None else ""
                if not has_tqdm:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_str}")
                else:
                    tqdm.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_str}")

        return history

    def predict(self, X):
        """Run forward pass and return predictions as raw float lists."""
        predictions = []
        for x in X:
            x_val = [Value(xi) if not isinstance(xi, Value) else xi for xi in x]
            pred = self(x_val)
            pred_list = pred if isinstance(pred, list) else [pred]
            predictions.append([p.data for p in pred_list])
        return predictions

    def save(self, filepath):
        """Save model architecture and weights to a JSON file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations_list': self.activations_list,
            'init_method': self.init_method,
            'init_params': self.init_params,
            'weights': []
        }

        for layer in self.layers:
            layer_weights = []
            for neuron in layer.neurons:
                neuron_data = {
                    'w': [w.data for w in neuron.w],
                    'b': neuron.b.data
                }
                layer_weights.append(neuron_data)
            model_data['weights'].append(layer_weights)

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load model from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        model = cls(
            layer_sizes=model_data['layer_sizes'],
            activations_list=model_data['activations_list'],
            init_method=model_data.get('init_method', 'uniform'),
            init_params=model_data.get('init_params', None)
        )

        # Restore weights
        for layer, layer_weights in zip(model.layers, model_data['weights']):
            for neuron, neuron_data in zip(layer.neurons, layer_weights):
                for w, w_val in zip(neuron.w, neuron_data['w']):
                    w.data = w_val
                neuron.b.data = neuron_data['b']

        return model

    def plot_weight_distribution(self, layer_indices=None, save_path=None):
        """
        Plot histogram of weights for specified layers.

        Parameters:
        - layer_indices: list of int, which layers to plot (0-indexed). None = all layers.
        - save_path: if provided, save figure to this path instead of showing.
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        n_plots = len(layer_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, idx in zip(axes, layer_indices):
            if idx < 0 or idx >= len(self.layers):
                continue
            weights = [w.data for w in self.layers[idx].weights()]
            ax.hist(weights, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f"Layer {idx} Weights\n({self.layers[idx].activation_name})")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Count")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_gradient_distribution(self, layer_indices=None, save_path=None):
        """
        Plot histogram of gradients for specified layers.
        Call after backward() to see meaningful gradients.

        Parameters:
        - layer_indices: list of int, which layers to plot (0-indexed). None = all layers.
        - save_path: if provided, save figure to this path instead of showing.
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        n_plots = len(layer_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, idx in zip(axes, layer_indices):
            if idx < 0 or idx >= len(self.layers):
                continue
            grads = [p.grad for p in self.layers[idx].parameters()]
            ax.hist(grads, bins=30, alpha=0.7, edgecolor='black', color='coral')
            ax.set_title(f"Layer {idx} Gradients\n({self.layers[idx].activation_name})")
            ax.set_xlabel("Gradient Value")
            ax.set_ylabel("Count")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()