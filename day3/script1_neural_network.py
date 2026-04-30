# ============================================================
# DAY 3 — SCRIPT 1: Neural Networks with TensorFlow & Keras
# Topics: perceptrons, activation functions, forward pass,
#         backprop, building & training a neural net on MNIST
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# TensorFlow import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("DAY 3 — SCRIPT 1: Neural Networks with TensorFlow")
print(f"TensorFlow version: {tf.__version__}")
print("=" * 60)


# ── SECTION 1: What IS a Neural Network? ─────────────────────
print("\nSECTION 1: Neural Network Intuition")
print("-" * 40)
print("""
A neural network is just MATH stacked in layers:

  Input Layer → Hidden Layer(s) → Output Layer

Each connection has a WEIGHT (w) and each neuron has a BIAS (b).
The neuron computes: output = activation(w * input + b)

That's it. Everything else is just this repeated millions of times.
""")


# ── SECTION 2: Manual Neuron from Scratch ─────────────────────
print("SECTION 2: Building a Single Neuron Manually")
print("-" * 40)

def sigmoid(x):
    """Squashes any number to range (0, 1) — used in output layer"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Returns 0 if negative, x if positive — most common hidden layer activation"""
    return np.maximum(0, x)

def tanh(x):
    """Squashes to range (-1, 1)"""
    return np.tanh(x)

# Simulate a single neuron
inputs  = np.array([0.5, 0.3, 0.8])   # 3 input features
weights = np.array([0.4, 0.7, 0.2])   # learned during training
bias    = 0.1

# Forward pass of ONE neuron
z      = np.dot(inputs, weights) + bias   # weighted sum
output_sigmoid = sigmoid(z)
output_relu    = relu(z)

print(f"Inputs   : {inputs}")
print(f"Weights  : {weights}")
print(f"Bias     : {bias}")
print(f"z = dot(inputs, weights) + bias = {z:.4f}")
print(f"Sigmoid(z) = {output_sigmoid:.4f}  → probability output")
print(f"ReLU(z)    = {output_relu:.4f}  → hidden layer output")

# Show activation functions visually
x_vals = np.linspace(-5, 5, 100)
print("\nActivation function ranges:")
print(f"  Sigmoid: {sigmoid(-5):.3f} to {sigmoid(5):.3f}  → use for binary output")
print(f"  ReLU   : {relu(-5):.3f} to {relu(5):.3f}  → use for hidden layers")
print(f"  Tanh   : {tanh(-5):.3f} to {tanh(5):.3f}  → alternative to sigmoid")


# ── SECTION 3: Manual 2-Layer Network ─────────────────────────
print("\n\nSECTION 3: Manual 2-Layer Neural Network (Forward Pass)")
print("-" * 40)
print("Predicting if a student gets placed based on 3 features")

np.random.seed(42)

# Network architecture: 3 → 4 → 1
# Input: [cgpa_normalized, internships_normalized, projects_normalized]
# Output: placement probability

# Layer 1 weights & biases (3 inputs → 4 neurons)
W1 = np.random.randn(3, 4) * 0.1
b1 = np.zeros(4)

# Layer 2 weights & biases (4 neurons → 1 output)
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros(1)

# One student's data (normalized)
student = np.array([0.85, 0.67, 0.90])  # cgpa=8.5/10, 2 internships, 4-5 projects

# Forward pass
z1 = student @ W1 + b1     # hidden layer weighted sum
a1 = relu(z1)               # hidden layer activation
z2 = a1 @ W2 + b2           # output layer weighted sum
a2 = sigmoid(z2)            # output — probability of placement

print(f"Student features (normalized): {student}")
print(f"\nLayer 1 (hidden):")
print(f"  z1 = {z1.round(4)}")
print(f"  a1 = ReLU(z1) = {a1.round(4)}")
print(f"\nLayer 2 (output):")
print(f"  z2 = {z2.round(4)}")
print(f"  a2 = Sigmoid(z2) = {a2[0]:.4f}")
print(f"\nPlacement probability: {a2[0]:.2%}")
print(f"Prediction: {'PLACED ✓' if a2[0] > 0.5 else 'NOT PLACED ✗'}")
print("\n(Note: weights are random here — not trained yet. Accuracy will be bad.)")
print("Training with backpropagation fixes the weights automatically.")


# ── SECTION 4: MNIST Dataset ─────────────────────────────────
print("\n\nSECTION 4: Loading the MNIST Dataset")
print("-" * 40)
print("MNIST = 70,000 images of handwritten digits (0-9)")
print("This is the 'Hello World' of deep learning\n")

# Load dataset — downloads automatically first time (~11MB)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training images : {X_train.shape}  → 60,000 images of 28x28 pixels")
print(f"Training labels : {y_train.shape}  → the digit each image shows")
print(f"Test images     : {X_test.shape}")
print(f"Test labels     : {y_test.shape}")
print(f"\nPixel value range: {X_train.min()} to {X_train.max()}")
print(f"Label classes   : {np.unique(y_train)}")

# Show distribution
print(f"\nDigit distribution in training set:")
for digit in range(10):
    count = (y_train == digit).sum()
    bar   = "█" * (count // 300)
    print(f"  {digit}: {count}  {bar}")


# ── SECTION 5: Preprocess Data ───────────────────────────────
print("\n\nSECTION 5: Preprocessing")
print("-" * 40)

# Normalize: pixel values 0-255 → 0.0-1.0
# WHY: neural networks train much faster with small numbers
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# Flatten: 28x28 image → 784 values (one long array)
# WHY: Dense layers need a 1D input
X_train_flat = X_train.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

print(f"Before flatten : {(60000, 28, 28)}")
print(f"After flatten  : {X_train_flat.shape}")
print(f"Pixel range    : {X_train_flat.min():.1f} to {X_train_flat.max():.1f}  (normalized)")


# ── SECTION 6: Build the Model ───────────────────────────────
print("\n\nSECTION 6: Building the Neural Network")
print("-" * 40)

model = keras.Sequential([
    # Input layer — 784 pixels coming in
    layers.Input(shape=(784,)),

    # Hidden layer 1 — 128 neurons, ReLU activation
    layers.Dense(128, activation="relu"),

    # Dropout — randomly turns off 20% of neurons during training
    # WHY: prevents overfitting (memorizing instead of learning)
    layers.Dropout(0.2),

    # Hidden layer 2 — 64 neurons
    layers.Dense(64, activation="relu"),

    layers.Dropout(0.2),

    # Output layer — 10 neurons (one per digit 0-9)
    # Softmax converts to probabilities that sum to 1
    layers.Dense(10, activation="softmax")
])

model.summary()

print(f"\nTotal parameters: {model.count_params():,}")
print("These are the weights that get adjusted during training!")


# ── SECTION 7: Compile & Train ───────────────────────────────
print("\n\nSECTION 7: Compiling & Training")
print("-" * 40)
print("Compile = tell the model HOW to train")
print("  optimizer : how to update weights (Adam = most popular)")
print("  loss      : what to minimize (cross-entropy for classification)")
print("  metrics   : what to track (accuracy)")

model.compile(
    optimizer = "adam",
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
)

print("\nTraining started... (5 epochs = 5 passes through all 60,000 images)")

history = model.fit(
    X_train_flat, y_train,
    epochs          = 5,
    batch_size      = 32,       # process 32 images at a time
    validation_split= 0.1,     # use 10% of train data for validation
    verbose         = 1
)


# ── SECTION 8: Evaluate ───────────────────────────────────────
print("\n\nSECTION 8: Evaluating on Test Data")
print("-" * 40)

test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Test Accuracy : {test_acc:.2%}")
print(f"Test Loss     : {test_loss:.4f}")
print(f"\nThat means the model correctly identifies {test_acc*100:.1f}% of handwritten digits!")


# ── SECTION 9: Make Predictions ──────────────────────────────
print("\n\nSECTION 9: Making Predictions")
print("-" * 40)

predictions = model.predict(X_test_flat[:10], verbose=0)

print(f"{'Sample':<8} {'Actual':>7} {'Predicted':>10} {'Confidence':>12} {'Correct?':>9}")
print("─" * 50)
for i in range(10):
    actual    = y_test[i]
    predicted = np.argmax(predictions[i])
    confidence= np.max(predictions[i])
    correct   = "✓" if actual == predicted else "✗"
    print(f"{i+1:<8} {actual:>7} {predicted:>10} {confidence:>11.2%} {correct:>9}")


# ── SECTION 10: Visualise ─────────────────────────────────────
print("\n\nSECTION 10: Saving visualisation")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Day 3 — Neural Network on MNIST", fontsize=14, fontweight="bold")

BLUE = "#378ADD"; GREEN = "#1D9E75"; ORANGE = "#BA7517"

# Chart 1: Training accuracy curve
ax1 = axes[0, 0]
ax1.plot(history.history["accuracy"],     color=BLUE,   linewidth=2, label="Train accuracy")
ax1.plot(history.history["val_accuracy"], color=ORANGE, linewidth=2, linestyle="--", label="Val accuracy")
ax1.set_title("Accuracy over Epochs"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.set_ylim(0, 1)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# Chart 2: Training loss curve
ax2 = axes[0, 1]
ax2.plot(history.history["loss"],     color=BLUE,   linewidth=2, label="Train loss")
ax2.plot(history.history["val_loss"], color=ORANGE, linewidth=2, linestyle="--", label="Val loss")
ax2.set_title("Loss over Epochs"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend()
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# Chart 3: Activation functions
ax3 = axes[0, 2]
x_plot = np.linspace(-4, 4, 200)
ax3.plot(x_plot, sigmoid(x_plot), color=BLUE,   linewidth=2, label="Sigmoid")
ax3.plot(x_plot, relu(x_plot),    color=GREEN,  linewidth=2, label="ReLU")
ax3.plot(x_plot, tanh(x_plot),    color=ORANGE, linewidth=2, label="Tanh")
ax3.axhline(0, color="gray", linewidth=0.5); ax3.axvline(0, color="gray", linewidth=0.5)
ax3.set_title("Activation Functions"); ax3.legend()
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# Charts 4-6: Sample predictions
for i, ax in enumerate(axes[1]):
    img = X_test[i].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    pred  = np.argmax(predictions[i])
    conf  = np.max(predictions[i])
    color = "green" if pred == y_test[i] else "red"
    ax.set_title(f"Actual: {y_test[i]}  Pred: {pred}\nConf: {conf:.1%}", color=color, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("day3_neural_network.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart saved as 'day3_neural_network.png'")

print()
print("=" * 60)
print(f"Script 1 complete! Neural network trained to {test_acc:.2%} accuracy!")
print("Key concepts covered:")
print("  ✓ Neurons, weights, biases, activation functions")
print("  ✓ Forward pass (manual + with Keras)")
print("  ✓ MNIST dataset loading & preprocessing")
print("  ✓ Building a Sequential model")
print("  ✓ Dropout for regularization")
print("  ✓ Training with .fit()")
print("  ✓ Evaluating with .evaluate()")
print("=" * 60)
