# IMPORTS
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from IPython.display import clear_output
tf.config.run_functions_eagerly(True)

# Training parameters
learning_rate = 0.0001
batch_size = 32
epochs = 25000

"""
Be prepared to change this since we are going to be working with audio samples, not images
"""
# Network parameters
image_dimension = 784  # Image size is 28 x 28

# Discriminator nodes
"""
This hidden dimension dictates that we are making a neural network
with a hidden dimension with 128 nodes. This helps in creating
the architecture of the neural networks used for discrimination and
generation

For the Discriminators, this is the architecture of the neural network

              INPUT         HIDDEN LAYER          OUTPUT
            784 Nodes         128 Nodes           1 Node

For the Generators, this is the architecture of the neural networks

              INPUT         HIDDEN LAYER          OUTPUT
            784 Nodes         128 Nodes           784 Nodes*


We have 784 input nodes since we are taking in the pixels of the image,
similarly to a handwritten digit classifier. Similarly, the 784 output values
correspond to every pixel that makes up the generated output image.

For an actual implementation, we would want to make this architecture more
advanced using a CNN and not a neural network with one layer
"""
H_dim = 128

# Weight initialization function
"""
This function is what we are using in order to initialize the Weights and
Biases for our Discriminator and our Generator.
"""
def xavier_init(shape):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.Variable(initializer(shape=shape), trainable=True)

# Placeholder variables
"""
CycleGAN requires two sets of inputs from the outside in order to train on
which will correspond to the placeholders that we see below. Initializing
the shape with None for the amount of examples because we do not know how many
inputs we are going to be receiving from our placeholders. However, we do know
that each example is going to have 784 features corresponding to the amount
of pixels in the image
"""
X_A = tf.Variable(tf.random.normal([batch_size, image_dimension], dtype=tf.float32))
X_B = tf.Variable(tf.random.normal([batch_size, image_dimension], dtype=tf.float32))

# Define weight and bias dictionaries for Discriminators and Generators
Disc_A_W = {
    "disc_H": tf.keras.layers.Dense(H_dim, activation=None, use_bias=False, input_shape=(image_dimension,)),
    "disc_final": tf.keras.layers.Dense(1, activation=None, use_bias=True)
}
Disc_A_B = {"disc_H": xavier_init([H_dim]), "disc_final": xavier_init([1])}
Disc_B_W = {
    "disc_H": tf.keras.layers.Dense(H_dim, activation=None, use_bias=False, input_shape=(image_dimension,)),
    "disc_final": tf.keras.layers.Dense(1, activation=None, use_bias=True)
}
Disc_B_B = {"disc_H": xavier_init([H_dim]), "disc_final": xavier_init([1])}
Gen_AB_W = {"Gen_H": xavier_init([image_dimension, H_dim]), "Gen_final": xavier_init([H_dim, image_dimension])}
Gen_AB_B = {"Gen_H": xavier_init([H_dim]), "Gen_final": xavier_init([image_dimension])}
Gen_BA_W = {"Gen_H": xavier_init([image_dimension, H_dim]), "Gen_final": xavier_init([H_dim, image_dimension])}
Gen_BA_B = {"Gen_H": xavier_init([H_dim]), "Gen_final": xavier_init([image_dimension])}


# Define Discriminator A as a model
Disc_A_model = tf.keras.Sequential([
    tf.keras.layers.Dense(H_dim, activation='leaky_relu', input_shape=(image_dimension,), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define Discriminator B as a model
Disc_B_model = tf.keras.Sequential([
    tf.keras.layers.Dense(H_dim, activation='leaky_relu', input_shape=(image_dimension,), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Define network functions
def Disc_A(x):
    x = tf.cast(x, tf.float32)
    return Disc_A_model(x)

def Disc_B(x):
    x = tf.cast(x, tf.float32)
    return Disc_B_model(x)



def Gen_AB(x):
    x = tf.cast(x, tf.float32)
    hidden_layer_pred = tf.nn.leaky_relu(tf.add(tf.matmul(x, Gen_AB_W["Gen_H"]), Gen_AB_B["Gen_H"]))
    hidden_layer_pred = tf.keras.layers.BatchNormalization()(hidden_layer_pred)
    output_layer_pred = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_pred, Gen_AB_W["Gen_final"]), Gen_AB_B["Gen_final"]))
    return output_layer_pred

def Gen_BA(x):
    x = tf.cast(x, tf.float32)
    hidden_layer_pred = tf.nn.leaky_relu(tf.add(tf.matmul(x, Gen_BA_W["Gen_H"]), Gen_BA_B["Gen_H"]))
    hidden_layer_pred = tf.keras.layers.BatchNormalization()(hidden_layer_pred)
    output_layer_pred = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_pred, Gen_BA_W["Gen_final"]), Gen_BA_B["Gen_final"]))
    return output_layer_pred

# CycleGAN Setup
"""
Now let us actually build the CycleGAN Network

First, we begin by creating the GAN (Generative Adversarial Network)
for approximating A's distribution. In order to do this, we use B to A
Generator that we created, trained on input B. Then, we train the
Discriminator on real input from input A as well as fake input from the
B to A input we generated.
"""
X_BA = Gen_BA(X_B)
Disc_A_real = Disc_A(X_A)
Disc_A_fake = Disc_A(X_BA)
"""
Then, we create the GAN (Generative Adversarial Network) for approximating
B's distribution. In order to do this, we use A to B Generator that we
created, trained on input A. Then, we train the Discriminator on real input
from input B as well as fake input from the A to B input we generated.
"""
X_AB = Gen_AB(X_A)
Disc_B_real = Disc_B(X_B)
Disc_B_fake = Disc_B(X_AB)

# Discriminator and Generator losses
Disc_Loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_A_real, labels=tf.ones_like(Disc_A_real))) +
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_A_fake, labels=tf.zeros_like(Disc_A_fake))) +
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_B_real, labels=tf.ones_like(Disc_B_real))) +
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_B_fake, labels=tf.zeros_like(Disc_B_fake))))

Gen_Loss = (tf.reduce_mean(tf.square(Disc_B_fake - tf.ones_like(Disc_B_fake))) +
            tf.reduce_mean(tf.square(Disc_A_fake - tf.ones_like(Disc_A_fake))) +
            tf.reduce_mean(10 * tf.abs(X_A - Gen_BA(X_AB))) +
            tf.reduce_mean(10 * tf.abs(X_B - Gen_AB(X_BA))))

# Define optimizers
Gen_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
Disc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train_step(X_A_batch, X_B_batch):
    X_A_batch = tf.cast(X_A_batch, tf.float32)
    X_B_batch = tf.cast(X_B_batch, tf.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        X_BA = Gen_BA(X_B_batch)
        X_AB = Gen_AB(X_A_batch)

        Disc_A_real = Disc_A(X_A_batch)
        Disc_A_fake = Disc_A(X_BA)
        Disc_B_real = Disc_B(X_B_batch)
        Disc_B_fake = Disc_B(X_AB)

        Disc_Loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_A_real, labels=tf.ones_like(Disc_A_real))) +
                     tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_A_fake, labels=tf.zeros_like(Disc_A_fake))) +
                     tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_B_real, labels=tf.ones_like(Disc_B_real))) +
                     tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_B_fake, labels=tf.zeros_like(Disc_B_fake))))

        X_BAB = Gen_AB(X_BA)
        X_ABA = Gen_BA(X_AB)
        
        Gen_Loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_B_fake, labels=tf.ones_like(Disc_B_fake))) +
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Disc_A_fake, labels=tf.ones_like(Disc_A_fake))) +
                    tf.reduce_mean(10 * tf.abs(X_A_batch - X_ABA)) +
                    tf.reduce_mean(10 * tf.abs(X_B_batch - X_BAB)))

    # Use trainable_variables to compute gradients
    Gen_gradients = gen_tape.gradient(Gen_Loss, list(Gen_AB_W.values()) + list(Gen_AB_B.values()) + list(Gen_BA_W.values()) + list(Gen_BA_B.values()))
    Disc_gradients = disc_tape.gradient(Disc_Loss, Disc_A_model.trainable_variables + Disc_B_model.trainable_variables)

    # Apply gradients
    Gen_optimizer.apply_gradients(zip(Gen_gradients, list(Gen_AB_W.values()) + list(Gen_AB_B.values()) + list(Gen_BA_W.values()) + list(Gen_BA_B.values())))
    Disc_optimizer.apply_gradients(zip(Disc_gradients, Disc_A_model.trainable_variables + Disc_B_model.trainable_variables))

    return Disc_Loss, Gen_Loss


# Loading the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preparing datasets
mid = int(X_train.shape[0] / 2)
X_train_real = X_train[:mid].reshape(-1, 784).astype(np.float32) / 255.0
X_train_rot = scipy.ndimage.rotate(X_train[mid:].reshape(-1, 28, 28), 90, axes=(1, 2)).reshape(-1, 784).astype(np.float32) / 255.0

# Display function
def display_samples(X_A_batch, X_B_batch, n=6):
    canvas1 = np.empty((28 * n, 28 * n))
    canvas2 = np.empty((28 * n, 28 * n))

    for i in range(n):
        out_A = Gen_BA(X_B_batch).numpy()
        out_B = Gen_AB(X_A_batch).numpy()

        for j in range(n):
            canvas1[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = out_A[j].reshape([28, 28])
            canvas2[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = out_B[j].reshape([28, 28])

    plt.figure(figsize=(n, n))
    plt.imshow(canvas1, origin="upper", cmap="gray")
    plt.show()

    plt.figure(figsize=(n, n))
    plt.imshow(canvas2, origin="upper", cmap="gray")
    plt.show()

# Training loop
for epoch in range(epochs):
    X_A_batch = X_train_real[np.random.choice(X_train_real.shape[0], batch_size, replace=False)]
    X_B_batch = X_train_rot[np.random.choice(X_train_rot.shape[0], batch_size, replace=False)]

    Disc_loss, Gen_loss = train_step(X_A_batch, X_B_batch)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {Disc_loss}, Generator Loss: {Gen_loss}")
        display_samples(X_A_batch, X_B_batch)


display_samples(X_train_real[:batch_size], X_train_rot[:batch_size])
