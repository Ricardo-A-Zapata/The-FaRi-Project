import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import os
import shutil
from IPython.display import clear_output
tf.config.run_functions_eagerly(True)

"""
Training Parameters/Constants:
    - LEARN_RATE: Learning rate for the optimizers
    - BATCH_SIZE: Number of samples processed in one batch
    - EPOCHS: Number of training iterations
    - OUTPUT_DIR: Directory for saving generated outputs

Input Dimensions for Model Architecture:

    NOTICE: These dimensions are for image-based input
    (28 x 28 images flattened to 784 pixels). # When transitioning
    to spectrograms, these will need to match the spectrogram dimensions.

    - IMAGE_DIM: Flattened image dimensions (28 x 28)
    - HIDDEN_DIM: Number of nodes in fully connected layers (if used)


    HIDDEN_DIM Explanation:
    This parameter defines the size of the intermediate dense layers in
    the generator and discriminator architectures.

    In this architecture:
    1. Discriminators:
        - HIDDEN_DIM is used in the final fully connected layer to map
        features to a single output (real/fake classification).
    2. Generators:
        - HIDDEN_DIM applies to the intermediate dense layers that transform
        features before output.

Future Considerations:
For spectrograms, convolutional kernel sizes, strides, and the number of filters
should be adjusted based on the dimensions of the input data.
"""

LEARN_RATE = 0.0001
BATCH_SIZE = 32  
EPOCHS = 25000 
OUTPUT_DIR = "output"
IMAGE_DIM = 784 
HIDDEN_DIM = 128


"""
Placeholder Variables:
- X_A: Samples from domain A (unaltered images or spectrograms).
- X_B: Samples from domain B (rotated images or transformed spectrograms).

Each placeholder's shape includes:
1. Batch size (BATCH_SIZE) - the number of samples per batch.
2. Feature size (IMAGE_DIM) - 784 for flattened 28x28 images. For spectrograms,
   this would reflect the spectrogram dimensions.
"""
X_A = tf.Variable(tf.random.normal([BATCH_SIZE, IMAGE_DIM], dtype=tf.float32))
X_B = tf.Variable(tf.random.normal([BATCH_SIZE, IMAGE_DIM], dtype=tf.float32))


"""
CNN-Based Model Definitions:
1. Discriminators:
   - Extract features through convolutional layers.
   - Use a fully connected layer with HIDDEN_DIM to map features to a single
     output node (real/fake).

2. Generators:
   - Transform inputs between domains using convolutional layers.
   - Output is reshaped to match the input dimensions (e.g., flattened image
     or spectrogram size).

Activation Functions:
- Leaky ReLU: Prevents dead neurons during training.
- Sigmoid: Normalizes outputs for probabilities or pixel intensities.
- Batch Normalization: Stabilizes training and prevents mode collapse.

Future Enhancements:
For spectrograms, replace dense layers entirely with convolutional layers that
better capture temporal or frequency patterns.
"""
Disc_A = tf.keras.Sequential([
    tf.keras.Input(shape=(IMAGE_DIM,)),
    tf.keras.layers.Dense(HIDDEN_DIM, activation='leaky_relu', use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Disc_B = tf.keras.Sequential([
    tf.keras.Input(shape=(IMAGE_DIM,)),
    tf.keras.layers.Dense(HIDDEN_DIM, activation='leaky_relu', use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Gen_AB = tf.keras.Sequential([
    tf.keras.Input(shape=(IMAGE_DIM,)),
    tf.keras.layers.Dense(HIDDEN_DIM, activation='leaky_relu', use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(IMAGE_DIM, activation='sigmoid')
])

Gen_BA = tf.keras.Sequential([
    tf.keras.Input(shape=(IMAGE_DIM,)),
    tf.keras.layers.Dense(HIDDEN_DIM, activation='leaky_relu', use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(IMAGE_DIM, activation='sigmoid')
])


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




"""
CycleGAN Loss Functions:
1. Discriminator Loss (LSGAN):
   - Uses Mean Squared Error (MSE) loss.
   - Encourages the discriminator to classify real samples as close to
     1 and fake samples as close to 0.

2. Generator Loss:
   - Includes adversarial loss to fool the discriminators (MSE).
   - Incorporates cycle-consistency loss to ensure transformations retain
     the original input features when cycled back.
   - `lambda_cyc` controls the weight of the cycle-consistency loss.

Future Considerations:
For spectrogram data, perceptual loss (e.g., log-magnitude or spectral convergence) could replace or augment the cycle-consistency loss to better reflect audio fidelity.
"""

Disc_Loss = (
    tf.reduce_mean(tf.square(Disc_A_real - 1)) +  # Penalize real samples for not being classified as 1
    tf.reduce_mean(tf.square(Disc_A_fake)) +      # Penalize fake samples for being classified as real
    tf.reduce_mean(tf.square(Disc_B_real - 1)) +
    tf.reduce_mean(tf.square(Disc_B_fake))
)

lambda_cyc = 10 

Gen_Loss = (
    tf.reduce_mean(tf.square(Disc_B_fake - 1)) +  # Adversarial loss for B
    tf.reduce_mean(tf.square(Disc_A_fake - 1)) +  # Adversarial loss for A
    lambda_cyc * tf.reduce_mean(tf.abs(X_A - Gen_BA(X_AB))) +  # Cycle-consistency A -> B -> A
    lambda_cyc * tf.reduce_mean(tf.abs(X_B - Gen_AB(X_BA)))    # Cycle-consistency B -> A -> B
)

# Define optimizers
Gen_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
Disc_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)




"""
Training Step:
Performs a single update of the generator and discriminator parameters.

1. Loss Computation:
   - Discriminator loss evaluates real vs. fake classifications using MSE loss (LSGAN).
   - Generator loss includes adversarial loss (fooling discriminators) and cycle-consistency loss to ensure transformations retain input features.

2. Gradient Computation:
   - Computes gradients for both generator and discriminator models using `tf.GradientTape`.
   - Updates weights using the Adam optimizer.

Key Considerations:
Ensure that batch size, input size, and learning rate are tuned to match the domain-specific data (e.g., spectrograms for audio).
"""

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
    Gen_gradients = gen_tape.gradient(Gen_Loss, Gen_AB.trainable_variables + Gen_BA.trainable_variables)
    Disc_gradients = disc_tape.gradient(Disc_Loss, Disc_A.trainable_variables + Disc_B.trainable_variables)

    # Apply gradients
    Gen_optimizer.apply_gradients(zip(Gen_gradients, Gen_AB.trainable_variables + Gen_BA.trainable_variables))
    Disc_optimizer.apply_gradients(zip(Disc_gradients, Disc_A.trainable_variables + Disc_B.trainable_variables))

    return Disc_Loss, Gen_Loss

# Reset output directory
if os.path.exists(OUTPUT_DIR):
    if os.listdir(OUTPUT_DIR):
        print(f"\nClearing the directory: {OUTPUT_DIR}")
        print("-------------------------------------------")
        for file in os.listdir(OUTPUT_DIR):
            print(f"Removing: {file}")
        print("-------------------------------------------")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nOutput Directory created! Find model output in {OUTPUT_DIR} directory.\n")
print(f"Beginning model training...")
print("-------------------------------------------")

# Loading the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preparing datasets
mid = int(X_train.shape[0] / 2)
X_train_real = X_train[:mid].reshape(-1, 784).astype(np.float32) / 255.0
X_train_rot = scipy.ndimage.rotate(X_train[mid:].reshape(-1, 28, 28), 90, axes=(1, 2)).reshape(-1, 784).astype(np.float32) / 255.0


def display_samples(X_A_batch, X_B_batch, epoch, n=6):
    # Create the visualization grids
    canvas_AB = np.empty((28 * n, 28 * n))  # For Gen_AB (A -> B)
    canvas_BA = np.empty((28 * n, 28 * n))  # For Gen_BA (B -> A)

    # Generate images
    out_A = Gen_AB(X_A_batch).numpy()  # Generated B from A (A -> B)
    out_B = Gen_BA(X_B_batch).numpy()  # Generated A from B (B -> A)

    for i in range(n):
        for j in range(n):
            # Populate grid for Gen_AB (A -> B)
            canvas_AB[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = out_B[j].reshape([28, 28])
            # Populate grid for Gen_BA (B -> A)
            canvas_BA[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = out_A[j].reshape([28, 28])

    # Save the visualization for Gen A -> B
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_AB, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title(f"Epoch {epoch}: Gen A -> B", fontsize=14, pad=15)
    plt.savefig(os.path.join(OUTPUT_DIR, f"Epoch_{epoch:04d}_Gen_AB.png"))  # Use zero-padded numbers
    plt.close()

    # Save the visualization for Gen B -> A
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_BA, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title(f"Epoch {epoch}: Gen B -> A", fontsize=14, pad=15)
    plt.savefig(os.path.join(OUTPUT_DIR, f"Epoch_{epoch:04d}_Gen_BA.png"))  # Use zero-padded numbers
    plt.close()


# Training loop
for epoch in range(EPOCHS + 1):
    X_A_batch = X_train_real[np.random.choice(X_train_real.shape[0], BATCH_SIZE, replace=False)]
    X_B_batch = X_train_rot[np.random.choice(X_train_rot.shape[0], BATCH_SIZE, replace=False)]

    Disc_loss, Gen_loss = train_step(X_A_batch, X_B_batch)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {Disc_loss}, Generator Loss: {Gen_loss}")
        display_samples(X_A_batch, X_B_batch, epoch)