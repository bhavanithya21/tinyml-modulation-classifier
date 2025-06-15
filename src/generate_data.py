import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
import os

# Parameters
num_samples = 1000     # per class
classes = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM']
output_dir = 'data'

os.makedirs(output_dir, exist_ok=True)

def add_awgn_noise(signal_mod, snr_db):
    snr = 10**(snr_db / 10)
    power_signal = np.mean(np.abs(signal_mod)**2)
    noise_power = power_signal / snr
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal_mod.shape) + 1j*np.random.randn(*signal_mod.shape))
    return signal_mod + noise

def generate_bpsk(length=1024):
    bits = np.random.randint(0, 2, length)
    symbols = 2*bits - 1  # Map 0->-1, 1->1
    return symbols.astype(np.complex64)

def generate_qpsk(length=1024):
    bits = np.random.randint(0, 4, length)
    mapping = {
        0: 1+1j,
        1: -1+1j,
        2: -1-1j,
        3: 1-1j
    }
    symbols = np.array([mapping[b] for b in bits])
    symbols /= np.sqrt(2)  # Normalize power
    return symbols.astype(np.complex64)

def generate_8psk(length=1024):
    bits = np.random.randint(0, 8, length)
    angles = 2 * np.pi * bits / 8
    symbols = np.exp(1j * angles)
    return symbols.astype(np.complex64)

def generate_16qam(length=1024):
    bits = np.random.randint(0, 16, length)
    mapping = {
        0: (-3+3j), 1: (-3+1j), 2: (-3-3j), 3: (-3-1j),
        4: (-1+3j), 5: (-1+1j), 6: (-1-3j), 7: (-1-1j),
        8: (3+3j), 9: (3+1j), 10: (3-3j), 11: (3-1j),
        12: (1+3j), 13: (1+1j), 14: (1-3j), 15: (1-1j),
    }
    symbols = np.array([mapping[b] for b in bits])
    symbols /= np.sqrt(10)  # Normalize power
    return symbols.astype(np.complex64)

def generate_64qam(length=1024):
    M = 8
    bits = np.random.randint(0, 64, length)
    # 64QAM const points from -7 to +7 spaced by 2 on I and Q
    I = 2*(bits % M) - (M-1)
    Q = 2*(bits // M) - (M-1)
    symbols = I + 1j*Q
    symbols /= np.sqrt((2/3)*(M**2 -1))  # Normalize power for M-QAM
    return symbols.astype(np.complex64)

def generate_pam4(length=1024):
    bits = np.random.randint(0, 4, length)
    mapping = {
        0: -3,
        1: -1,
        2: 3,
        3: 1
    }
    symbols = np.array([mapping[b] for b in bits], dtype=np.float32)
    symbols /= np.sqrt(5)  # Normalize power
    return symbols

def generate_gfsk(length=1024, bt=0.3, freq_dev=0.5, sampling_rate=50):
    # Approximate GFSK by shaping random bits with Gaussian filter and FSK modulation
    bits = np.random.randint(0, 2, length)
    bt_product = bt * sampling_rate  # bandwidth-time product related param
    # Gaussian filter impulse response
    gauss_filter = sig.windows.gaussian(M=4, std=bt_product)
    shaped_bits = sig.lfilter(gauss_filter, 1.0, bits*2-1)
    phase = 2 * np.pi * freq_dev * np.cumsum(shaped_bits) / sampling_rate
    signal_mod = np.exp(1j*phase)
    return signal_mod.astype(np.complex64)

def generate_cpfski(length=1024, freq_dev=0.5, sampling_rate=50):
    # Similar to GFSK, but continuous phase FSK simpler approx
    bits = np.random.randint(0, 2, length)
    phase = 2 * np.pi * freq_dev * np.cumsum(bits*2-1) / sampling_rate
    signal_mod = np.exp(1j*phase)
    return signal_mod.astype(np.complex64)

def generate_b_fm(length=1024, carrier_freq=10, sampling_rate=50, freq_dev=5):
    t = np.arange(length) / sampling_rate
    message = np.sin(2*np.pi*1*t)
    integral = np.cumsum(message) / sampling_rate
    signal_mod = np.cos(2*np.pi*carrier_freq*t + 2*np.pi*freq_dev*integral)
    return signal_mod.astype(np.float32)

def generate_dsb_am(length=1024, carrier_freq=10, sampling_rate=50):
    t = np.arange(length) / sampling_rate
    message = np.sin(2*np.pi*1*t)
    carrier = np.cos(2*np.pi*carrier_freq*t)
    signal_mod = (1 + message) * carrier
    return signal_mod.astype(np.float32)

def generate_ssb_am(length=1024, carrier_freq=10, sampling_rate=50):
    t = np.arange(length) / sampling_rate
    message = np.sin(2*np.pi*1*t)
    analytic_signal = sig.hilbert(message)
    carrier = np.exp(1j*2*np.pi*carrier_freq*t)
    ssb = np.real(analytic_signal * carrier)
    return ssb.astype(np.float32)

def plot_constellation(signal_mod, title="Constellation Diagram"):
    plt.figure(figsize=(6,6))
    plt.plot(signal_mod.real, signal_mod.imag, 'o', markersize=3, alpha=0.6)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.axis('equal')
    plt.show()

# Map class names to generator functions
modulation_generators = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "8PSK": generate_8psk,
    "16QAM": generate_16qam,
    "64QAM": generate_64qam,
    "PAM4": generate_pam4,
    "GFSK": generate_gfsk,
    "CPFSK": generate_cpfski,
    "B-FM": generate_b_fm,
    "DSB-AM": generate_dsb_am,
    "SSB-AM": generate_ssb_am,
}

# Generate data
X = []
Y = []

for idx, label in enumerate(classes):
    print(f"Generating {label} signals...")
    for _ in range(num_samples):
        mod_signal = modulation_generators[label]()
        X.append(mod_signal)
        Y.append(idx)

# Complex signals -- split into real and imaginary channels
if np.iscomplexobj(X):
    print('complex')
    X_real = np.real(X)
    X_imag = np.imag(X)
    X = np.stack([X_real, X_imag], axis=-1)  # shape: (samples, 1024, 2)
else:
    X_real = X
    X_imag = np.zeros_like(X_real)
    X = np.stack([X_real, X_imag], axis=-1)  # shape: (samples, 1024, 2)

X = np.array(X)
Y = np.array(Y)

np.save(os.path.join(output_dir, 'signals.npy'), X)
np.save(os.path.join(output_dir, 'labels.npy'), Y)

print("Signal data generated and saved.")

# Load and Plot

X_loaded = np.load("data/signals.npy")
Y_loaded = np.load("data/labels.npy")

print(X_loaded.shape)
print(X_real.shape)

num_classes = len(classes)
cols = 4
rows = math.ceil(num_classes / cols)
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True)

for idx, label in enumerate(classes):
    example_signal = X_loaded[Y_loaded == idx][0]
    row = idx // cols
    col = idx % cols    
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.plot(example_signal)
    ax.set_title(label)
    ax.grid(True)

# hide unused plots
for idx in range(num_classes, rows * cols):
    row = idx // cols
    col = idx % cols
    ax = axes[row, col] if rows > 1 else axes[col]
    ax.axis('off')

plt.tight_layout()
plt.show()

## Debug

signal_mod = generate_8psk(length=1024)
plot_constellation(signal_mod, title="QPSK Constellation")