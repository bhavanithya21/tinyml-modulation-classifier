import numpy as np
import tensorflow as tf
import scipy.signal as sig

# Functions

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

def run_inference(modulation_type, length=1024):

    if modulation_type not in mod_generators:
        raise ValueError(f"Modulation '{modulation_type}' not supported.")
    signal_mod = mod_generators[modulation_type](length)

    if np.iscomplexobj(signal_mod):
        signal_real = np.real(signal_mod)
        signal_imag = np.imag(signal_mod)
        input_data = np.stack((signal_real, signal_imag), axis=-1)
    else:
        # For real signals like B-FM, DSB-AM, SSB-AM
        signal_real = signal_mod
        signal_imag = np.zeros_like(signal_real)        
        input_data = np.stack((signal_real, signal_imag), axis=-1)
    
    print(input_data.shape)
    input_data = input_data / np.max(np.abs(input_data), axis=(0,1), keepdims=True)
    input_data = input_data[np.newaxis, ...].astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor (probabilities)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class index and name
    predicted_class = np.argmax(output_data)
    print(f"Output probabilities: {output_data}") 
    print(f"Predicted class: {classes[predicted_class]}")
       

mod_generators = {
    'BPSK': generate_bpsk,
    'QPSK': generate_qpsk,
    '8PSK': generate_8psk,
    '16QAM': generate_16qam,
    '64QAM': generate_64qam,
    'PAM4': generate_pam4,
    'GFSK': generate_gfsk,
    'CPFSK': generate_cpfski,
    'B-FM': generate_b_fm,
    'DSB-AM': generate_dsb_am,
    'SSB-AM': generate_ssb_am,
}

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/modulation_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define your classes (same order as training)
classes = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'GFSK', 'CPFSK', 'B-FM', 'DSB-AM', 'SSB-AM']

# Prepare input signal
mod_type = "PAM4"
run_inference(mod_type)
