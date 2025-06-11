import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import math
from tqdm import tqdm
import pickle

# Configuration
INPUT_DIM = 784
HIDDEN_DIM = 200
LATENT_DIM = 20
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

def init_linear(key, in_dim, out_dim):
    scale = math.sqrt(2.0 / in_dim)
    return jax.random.normal(key, (in_dim, out_dim)) * scale

def init_vae_params(key):
    keys = jax.random.split(key, 6)
    return {
        'encoder': {
            'w1': init_linear(keys[0], INPUT_DIM, HIDDEN_DIM),
            'b1': jnp.zeros(HIDDEN_DIM),
            'w_mu': init_linear(keys[1], HIDDEN_DIM, LATENT_DIM),
            'b_mu': jnp.zeros(LATENT_DIM),
            'w_logvar': init_linear(keys[2], HIDDEN_DIM, LATENT_DIM),
            'b_logvar': jnp.zeros(LATENT_DIM)
        },
        'decoder': {
            'w1': init_linear(keys[3], LATENT_DIM, HIDDEN_DIM),
            'b1': jnp.zeros(HIDDEN_DIM),
            'w2': init_linear(keys[4], HIDDEN_DIM, INPUT_DIM),
            'b2': jnp.zeros(INPUT_DIM)
        }
    }

def encode(params, x):
    h = jax.nn.relu(jnp.dot(x, params['w1']) + params['b1'])
    mu = jnp.dot(h, params['w_mu']) + params['b_mu']
    logvar = jnp.dot(h, params['w_logvar']) + params['b_logvar']
    return mu, logvar

def reparameterize(key, mu, logvar):
    eps = jax.random.normal(key, mu.shape)
    return mu + eps * jnp.exp(0.5 * logvar)

def decode(params, z):
    h = jax.nn.relu(jnp.dot(z, params['w1']) + params['b1'])
    return jax.nn.sigmoid(jnp.dot(h, params['w2']) + params['b2'])

def vae_forward(params, x, key):
    mu, logvar = encode(params['encoder'], x)
    z = reparameterize(key, mu, logvar)
    x_recon = decode(params['decoder'], z)
    return x_recon, mu, logvar

def vae_loss(params, x, key, beta=1.0):
    x_recon, mu, logvar = vae_forward(params, x, key)
    
    eps = 1e-8
    recon_loss = -jnp.mean(
        jnp.sum(x * jnp.log(x_recon + eps) + (1 - x) * jnp.log(1 - x_recon + eps), axis=1)
    )
    kl_loss = -0.5 * jnp.mean(
        jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1)
    )
    
    return recon_loss + beta * kl_loss

def load_mnist():

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)

    X = X.astype(np.float32) / 255.0
    
    return X, y

def create_batches(X, batch_size, key):
    n_samples = X.shape[0]
    indices = jax.random.permutation(key, n_samples)
    X_shuffled = X[indices]
    
    n_batches = n_samples // batch_size
    X_batches = X_shuffled[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    
    return X_batches

def make_train_step(optimizer):
    @jit
    def train_step(params, opt_state, batch, key):
        def loss_fn(params):
            return vae_loss(params, batch, key)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return train_step

def generate_digit(params, X, y, digit, num_examples=5, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    

    digit_idx = np.where(y == digit)[0][0]
    example = X[digit_idx:digit_idx+1]
    
    mu, logvar = encode(params['encoder'], example)
    
    generated_images = []
    keys = jax.random.split(key, num_examples)
    
    for i in range(num_examples):
        z = reparameterize(keys[i], mu, logvar)
        generated = decode(params['decoder'], z)
        generated_images.append(generated.reshape(28, 28))
    
    return generated_images

def save_generated_images(images, digit):
    for i, img in enumerate(images):
        plt.figure(figsize=(2, 2))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"generated_digit_{digit}_example_{i}.png", 
                   bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

def main():
    # Load data
    X, y = load_mnist()
    print(f"Loaded {X.shape[0]} samples")
    
    # Initialize
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key, 2)
    
    params = init_vae_params(init_key)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    

    train_step = make_train_step(optimizer)
    
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_key, train_key = jax.random.split(train_key, 2)
        X_batches = create_batches(X, BATCH_SIZE, epoch_key)
        
        epoch_losses = []
        
        pbar = tqdm(X_batches, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            batch_key, train_key = jax.random.split(train_key, 2)
            params, opt_state, loss = train_step(params, opt_state, batch, batch_key)
            
            epoch_losses.append(float(loss))
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    
    inference_key = jax.random.PRNGKey(123)
    
    for digit in range(10):
        digit_key, inference_key = jax.random.split(inference_key, 2)
        generated_images = generate_digit(params, X, y, digit, num_examples=3, key=digit_key)
        save_generated_images(generated_images, digit)
    
    return params

def save_model(params, filename="vae_params.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved as {filename}")

def load_model(filename="vae_params.pkl"):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f"Model loaded from {filename}")
    return params

if __name__ == "__main__":
    params = main()