# train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Load data
X = np.load("data/signals.npy")
Y = np.load("data/labels.npy")

# Normalize
X_Norm = X / np.max(np.abs(X), axis=(1, 2), keepdims=True)

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X_Norm, Y, test_size=0.2, random_state=42)

# Model definition
'''
model = models.Sequential([
    Input(shape=(1024, 2)),
    layers.Conv1D(16, 3, activation='relu', input_shape=(1024, 2)),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(Y)), activation='softmax')
])
'''

model = models.Sequential([
    Input(shape=(1024, 2)),  # Input layer, no need to specify input_shape in Conv1D
    
    layers.Conv1D(32, 5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(np.unique(Y)), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test))

# Save
os.makedirs("model", exist_ok=True)
model.save("model/modulation_classifier.keras")

# Extract losses and accuracies per epoch
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = len(train_acc)
batch_size = 32 
steps_per_epoch = len(X_train) // batch_size
total_iterations = epochs * steps_per_epoch

# To get iterations for plotting, we repeat each epoch metric for number of batches
iterations = np.arange(1, total_iterations + 1)

# Repeat each epoch metric 'steps_per_epoch' times to align with iterations
train_acc_iters = np.repeat(train_acc, steps_per_epoch)
val_acc_iters = np.repeat(val_acc, steps_per_epoch)
train_loss_iters = np.repeat(train_loss, steps_per_epoch)
val_loss_iters = np.repeat(val_loss, steps_per_epoch)

# If last epoch doesn't fill all batches exactly, truncate arrays to match iteration count
train_acc_iters = train_acc_iters[:total_iterations]
val_acc_iters = val_acc_iters[:total_iterations]
train_loss_iters = train_loss_iters[:total_iterations]
val_loss_iters = val_loss_iters[:total_iterations]

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Accuracy subplot
axs[0].plot(iterations, train_acc_iters, label='Training Accuracy')
axs[0].plot(iterations, val_acc_iters, label='Validation Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Accuracy vs Iterations')
axs[0].legend()
axs[0].grid(True)

# Loss subplot
axs[1].plot(iterations, train_loss_iters, label='Training Loss')
axs[1].plot(iterations, val_loss_iters, label='Validation Loss')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Loss')
axs[1].set_title('Loss vs Iterations')
axs[1].legend()
axs[1].grid(True)
plt.xlim(0, total_iterations - 1)  # make sure axis starts exactly at 0
plt.tight_layout()
plt.show()