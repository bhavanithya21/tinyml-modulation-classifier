import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
X = np.load("data/signals.npy") 
Y = np.load("data/labels.npy")

# Normalize data
X_Norm = X / np.max(np.abs(X), axis=(1, 2), keepdims=True)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_Norm, Y, test_size=0.2, random_state=42)

# Define model
model = models.Sequential([
    Input(shape=(1024, 2)),
    layers.Conv1D(16, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(Y)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# Predict on test set
Y_pred_probs = model.predict(X_test)
Y_pred = np.argmax(Y_pred_probs, axis=1)

# Generate confusion matrix
class_names = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM']
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
