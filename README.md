# WGAN-gp-ART
Using WGAN-gp and creating art portraits.
----
---

# Losses and Optimizers

This module defines a class for handling optimizers and loss functions for a Generative Adversarial Network (GAN) using TensorFlow and Keras.

## Requirements

- Python 3.x
- TensorFlow
- Keras

## Installation

To install the necessary libraries, you can use pip:

```sh
pip install tensorflow keras
```

## Usage

The `LossOptimizer` class provides methods to create and utilize optimizers and loss functions for training GANs.

### Initialization

Create an instance of the `LossOptimizer` class:

```python
from losses_optimizers import LossOptimizer

loss_optimizer = LossOptimizer()
```

### Optimizers

The `LossOptimizer` class initializes two optimizers with the Adam optimizer:
- `generator_optimizer`: Optimizer for the generator with a learning rate of 0.0001, beta_1 of 0.5, and beta_2 of 0.9.
- `discriminator_optimizer`: Optimizer for the discriminator with a learning rate of 0.0001, beta_1 of 0.5, and beta_2 of 0.9.

### Loss Functions

The class provides two static methods to compute the generator and discriminator losses:

- `generator_loss(fake_output)`: Computes the loss for the generator.
- `discriminator_loss(real_output, fake_output)`: Computes the loss for the discriminator.

### Example

Here is an example of how to use the `LossOptimizer` class:

```python
import tensorflow as tf
from losses_optimizers import LossOptimizer

# Create an instance of the LossOptimizer class
loss_optimizer = LossOptimizer()

# Generate some example data
real_output = tf.random.normal([1, 10])
fake_output = tf.random.normal([1, 10])

# Compute losses
gen_loss = loss_optimizer.generator_loss(fake_output)
disc_loss = loss_optimizer.discriminator_loss(real_output, fake_output)

print("Generator Loss:", gen_loss.numpy())
print("Discriminator Loss:", disc_loss.numpy())
```


---

---

# Generator and Discriminator Models

This module defines a class for creating the generator and discriminator models for a Generative Adversarial Network (GAN) using TensorFlow and Keras.

## Usage

The `Models` class provides methods to create the generator and discriminator models for training GANs.

### Initialization

Create an instance of the `Models` class:

```python
from models import Models

models = Models()
```

### Discriminator

The `discriminator` method builds and returns the discriminator model:

```python
discriminator_model = models.discriminator()
discriminator_model.summary()
```

The discriminator model consists of several convolutional layers, LeakyReLU activations, dropout layers, and layer normalization. It processes an input of shape `(128, 128, 3)` and outputs a single value indicating the realness of the input image.

### Generator

The `generator` method builds and returns the generator model:

```python
generator_model = models.generator()
generator_model.summary()
```

The generator model takes a latent vector of shape `(128,)` and transforms it through several dense and convolutional transpose layers with batch normalization and LeakyReLU activations, producing an output image of shape `(128, 128, 3)` with `tanh` activation.

### Example

Here is an example of how to use the `Models` class:

```python
from models import Models

# Create an instance of the Models class
models = Models()

# Build the discriminator model
discriminator = models.discriminator()

# Build the generator model
generator = models.generator()

# Print model summaries
discriminator.summary()
generator.summary()
```

---
Here is a README file that describes the `WGAN_gp` class and the `Monitor` callback:

---

# WGAN with Gradient Penalty and Training Monitor

This module defines a class for implementing Wasserstein GAN with Gradient Penalty (WGAN-GP) using TensorFlow and Keras, along with a custom callback for monitoring the training process.

## Requirements

- Python 3.x
- TensorFlow
- Keras

## Installation

To install the necessary libraries, you can use pip:

```sh
pip install tensorflow keras
```

## Usage

### WGAN_gp Class

The `WGAN_gp` class extends the Keras `Model` class to implement WGAN with Gradient Penalty.

#### Initialization

Create an instance of the `WGAN_gp` class by providing the generator and discriminator models along with the latent dimension size:

```python
from wgan_gp import WGAN_gp

# Assuming generator and discriminator models are already defined
wgan_gp = WGAN_gp(generator, discriminator, latent_dim=128)
```

#### Compilation

Compile the model by specifying the generator and discriminator loss functions and optimizers:

```python
wgan_gp.compile(
    g_loss=generator_loss,
    d_loss=discriminator_loss,
    g_opt=generator_optimizer,
    d_opt=discriminator_optimizer
)
```

#### Training

Train the model using the `fit` method:

```python
wgan_gp.fit(dataset, epochs=100, callbacks=[Monitor(num_img=5, latent_dim=128)])
```

### Monitor Class

The `Monitor` callback class saves generated images at the end of each epoch and logs the loss values.

#### Initialization

Create an instance of the `Monitor` class by specifying the number of images to generate and the latent dimension size:

```python
from wgan_gp import Monitor

monitor = Monitor(num_img=5, latent_dim=128)
```

#### Example

Here is an example of how to use both the `WGAN_gp` and `Monitor` classes together:

```python
from wgan_gp import WGAN_gp, Monitor
import tensorflow as tf

# Assuming generator and discriminator models, and loss functions and optimizers are defined
generator = ...
discriminator = ...
generator_loss = ...
discriminator_loss = ...
generator_optimizer = ...
discriminator_optimizer = ...

# Create WGAN-GP model
wgan_gp = WGAN_gp(generator, discriminator, latent_dim=128)

# Compile the model
wgan_gp.compile(
    g_loss=generator_loss,
    d_loss=discriminator_loss,
    g_opt=generator_optimizer,
    d_opt=discriminator_optimizer
)

# Create dataset (replace with actual dataset)
dataset = ...

# Initialize Monitor callback
monitor = Monitor(num_img=5, latent_dim=128)

# Train the model
wgan_gp.fit(dataset, epochs=100, callbacks=[monitor])
```

### Model Description

#### WGAN_gp Class

- **Initialization**: The class is initialized with a generator model, a discriminator model, and the latent dimension size. It also sets the gradient penalty weight and the number of extra discriminator steps.
- **Compilation**: The `compile` method sets the loss functions and optimizers for the generator and discriminator, and initializes metrics for tracking losses.
- **Gradient Penalty**: The `gradient_penalty` method computes the gradient penalty for a batch of real and fake images.
- **Training Step**: The `train_step` method performs a single training step, updating the discriminator multiple times for each update of the generator. It calculates losses, applies gradient penalties, and updates the model weights.

#### Monitor Class

- **Initialization**: The class is initialized with the number of images to generate and the latent dimension size.
- **Epoch End**: The `on_epoch_end` method generates images using the generator model and saves them to disk. It also logs the current epoch and loss values to a text file.


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- This module uses TensorFlow and Keras libraries for deep learning functionalities.
