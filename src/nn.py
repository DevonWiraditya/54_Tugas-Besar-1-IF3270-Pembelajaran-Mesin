import numpy as np
import functions as F
import json
import matplotlib.pyplot as plt

class RMSNorm:
    def __init__(self, size, epsilon=1e-8):
        self.epsilon = epsilon
        self.scale = F.Tensor(np.ones((1, size)))
        
    def parameters(self):
        return [self.scale]
        
    def forward(self, x):
        variance = (x ** 2).mean(axis=1, keepdims=True)
        x_norm = x / ((variance + self.epsilon) ** 0.5)
        return self.scale * x_norm

class Layer:
    def __init__(self, nin, nout, activation_name, init_method='uniform', init_params=None, seed=None, use_rmsnorm=False):
        self.nin = nin
        self.nout = nout
        self.activation_name = activation_name.lower()
        self.use_rmsnorm = use_rmsnorm
        if init_params is None: init_params = {}
        
        shape = (nin, nout)
        if init_method == 'zero':
            self.W = F.zero_init(shape)
            self.b = F.zero_init((1, nout))
        elif init_method == 'uniform':
            low = init_params.get('low', -1.0)
            high = init_params.get('high', 1.0)
            self.W = F.uniform_init(shape, low, high, seed)
            self.b = F.uniform_init((1, nout), low, high, seed)
        elif init_method == 'normal':
            mean = init_params.get('mean', 0.0)
            var = init_params.get('variance', 1.0)
            self.W = F.normal_init(shape, mean, var, seed)
            self.b = F.normal_init((1, nout), mean, var, seed)
            
        self.act_fn = F.get_activation(activation_name)
        self.rmsnorm = RMSNorm(nout) if use_rmsnorm else None
        
    def parameters(self):
        params = [self.W, self.b]
        if self.use_rmsnorm:
            params.extend(self.rmsnorm.parameters())
        return params

    def forward(self, X):
        Z = X @ self.W + self.b
        if self.use_rmsnorm:
            Z = self.rmsnorm.forward(Z)
        return self.act_fn(Z)

class FFNN:
    def __init__(self, layer_sizes, activations_list, init_method='uniform', init_params=None, seed=None, use_rmsnorm=False):
        self.layer_sizes = layer_sizes
        self.activations_list = activations_list
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            apply_norm = use_rmsnorm and (i < len(layer_sizes) - 2)
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations_list[i], init_method, init_params, seed, apply_norm))
            
    def forward(self, X):
        A = X if isinstance(X, F.Tensor) else F.Tensor(X, requires_grad=False)
        for layer in self.layers:
            A = layer.forward(A)
        return A
        
    def predict(self, X):
        return self.forward(X).data

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def fit(self, X_train, y_train, epochs, learning_rate, batch_size=1,
            loss_fn='mse', optimizer='sgd', X_val=None, y_val=None, verbose=1,
            reg_type=None, reg_lambda=0.0):
        
        history = {'train_loss': [], 'val_loss': []}
        
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        if optimizer == 'adam':
            adam_m = {id(p): np.zeros_like(p.data) for p in self.parameters()}
            adam_v = {id(p): np.zeros_like(p.data) for p in self.parameters()}
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            
        loss_funcs = {'mse': F.mse, 'bce': F.binary_crossentropy, 'cce': F.categorical_crossentropy}
        criterion = loss_funcs.get(loss_fn.lower(), F.mse)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if X_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

        m = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(m / batch_size))
            
            if verbose == 1 and has_tqdm:
                batch_iter = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            else:
                batch_iter = range(num_batches)
                
            for b in batch_iter:
                start = b * batch_size
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                Y_batch = y_shuffled[start:end]
                
                for p in self.parameters():
                    p.zero_grad()

                Y_pred = self.forward(X_batch)
                base_loss = criterion(Y_pred, Y_batch)
                
                base_loss.backward()
                
                if optimizer == 'adam':
                    t += 1

                for p in self.parameters():
                    grad = p.grad
                    if reg_type == 'l1':
                        grad += reg_lambda * np.sign(p.data)
                    elif reg_type == 'l2':
                        grad += 2 * reg_lambda * p.data
                        
                    if optimizer == 'adam':
                        pid = id(p)
                        m_w = adam_m[pid]
                        v_w = adam_v[pid]
                        
                        m_w = beta1 * m_w + (1 - beta1) * grad
                        v_w = beta2 * v_w + (1 - beta2) * (grad ** 2)
                        
                        adam_m[pid] = m_w
                        adam_v[pid] = v_w
                        
                        mw_hat = m_w / (1 - beta1**t)
                        vw_hat = v_w / (1 - beta2**t)
                        
                        p.data -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
                    else:
                        p.data -= learning_rate * grad

                epoch_loss += base_loss.data * (end - start)
                if verbose == 1 and has_tqdm and isinstance(batch_iter, tqdm):
                    batch_iter.set_postfix({'loss': f'{base_loss.data:.4f}'})

            train_loss = epoch_loss / m
            history['train_loss'].append(train_loss)
            
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_pred_tensor = F.Tensor(val_pred, requires_grad=False)
                val_base_loss = criterion(val_pred_tensor, y_val).data
                history['val_loss'].append(val_base_loss)
                if verbose == 1:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_base_loss:.4f}")
            else:
                history['val_loss'].append(None)
                if verbose == 1:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
                    
        return history

    def __call__(self, X):
        return self.forward(X)

    def save(self, filename):
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations_list,
            'weights': [layer.W.data.tolist() for layer in self.layers],
            'biases': [layer.b.data.tolist() for layer in self.layers]
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)
            
    @classmethod
    def load(cls, filename, use_rmsnorm=False):
        with open(filename, 'r') as f:
            md = json.load(f)
        model = cls(md['layer_sizes'], md['activations'], use_rmsnorm=use_rmsnorm)
        for layer, w, b in zip(model.layers, md['weights'], md['biases']):
            layer.W.data = np.array(w)
            layer.b.data = np.array(b)
        return model

    def plot_weight_distribution(self):
        ws = []
        for l in self.layers: ws.extend(l.W.data.flatten().tolist())
        plt.hist(ws, bins=50)
        plt.title('Weight Distribution')
        plt.show()

    def plot_gradient_distribution(self):
        gs = []
        for l in self.layers: gs.extend(l.W.grad.flatten().tolist())
        plt.hist(gs, bins=50)
        plt.title('Gradient Distribution')
        plt.show()