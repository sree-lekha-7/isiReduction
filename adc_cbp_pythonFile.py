import numpy as np
import matplotlib.pyplot as plt

# Parameters
symbol_rate = 1000  # Symbol rate in Hz
sampling_rate = 10 * symbol_rate  # Sampling rate in Hz
num_symbols = 100
bits_per_symbol = 4  # 16-QAM modulation

# Raised Cosine Filter
roll_off_factor = 0.35
filter_span = 8
t = np.arange(-filter_span, filter_span, 1.0 / sampling_rate)
raised_cosine_filter = np.sinc(t * symbol_rate) * np.cos(np.pi * roll_off_factor * t * symbol_rate) / (1 - (2 * roll_off_factor * t * symbol_rate) ** 2)

# QAM Modulation
bits = np.random.randint(0, 2, num_symbols * bits_per_symbol)
symbol_bits = bits.reshape(-1, bits_per_symbol)
symbol_indices = np.arange(0, 2 ** bits_per_symbol)
qam_constellation = np.exp(1j * 2 * np.pi * symbol_indices / 2 ** bits_per_symbol)

modulated_signal = np.zeros(num_symbols * sampling_rate, dtype=complex)
for i in range(num_symbols):
    symbol = np.dot(2 ** np.arange(bits_per_symbol - 1, -1, -1), symbol_bits[i])
    modulated_signal[i * sampling_rate:(i + 1) * sampling_rate] = np.sqrt(1 / 2) * qam_constellation[symbol]

# Add AWGN Channel Noise to Modulated Signal
snr_dB = 20  # Signal-to-noise ratio in dB
noise_power = 10 ** (-snr_dB / 10)  # Noise power based on SNR in dB
noise = np.sqrt(noise_power / 2) * (np.random.randn(len(modulated_signal)) + 1j * np.random.randn(len(modulated_signal)))
modulated_signal_with_noise = modulated_signal + noise

# Demodulation without Raised Cosine Filter
demodulated_signal_no_filter = np.zeros(num_symbols * bits_per_symbol, dtype=int)
for i in range(num_symbols):
    received_symbols = modulated_signal[i * sampling_rate + filter_span // 2:(i + 1) * sampling_rate: sampling_rate]
    distances = np.abs(received_symbols - qam_constellation)
    demodulated_symbol_index = np.argmin(distances)
    demodulated_signal_no_filter[i * bits_per_symbol:(i + 1) * bits_per_symbol] = np.array(
        [int(x) for x in np.binary_repr(demodulated_symbol_index, bits_per_symbol)])

# Demodulation with Raised Cosine Filter
filtered_signal = np.convolve(modulated_signal_with_noise, raised_cosine_filter, 'same')
demodulated_signal_with_filter = np.zeros(num_symbols * bits_per_symbol, dtype=int)
for i in range(num_symbols):
    received_symbols = filtered_signal[i * sampling_rate + filter_span // 2:(i + 1) * sampling_rate: sampling_rate]
    distances = np.abs(received_symbols - qam_constellation)
    demodulated_symbol_index = np.argmin(distances)
    demodulated_signal_with_filter[i * bits_per_symbol:(i + 1) * bits_per_symbol] = np.array(
        [int(x) for x in np.binary_repr(demodulated_symbol_index, bits_per_symbol)])

# Plot Constellation Diagram without Filter
plt.scatter(np.real(modulated_signal_with_noise), np.imag(modulated_signal_with_noise), marker='x', color='red', label='Received Symbols (No Filter)')
plt.scatter(np.real(modulated_signal), np.imag(modulated_signal), marker='o', label='Modulated Signal')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('QAM Modulation and Demodulation without Raised Cosine Filter')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot Constellation Diagram with Filter
plt.scatter(np.real(modulated_signal), np.imag(modulated_signal), marker='o', label='Modulated Signal')
plt.scatter(np.real(filtered_signal), np.imag(filtered_signal), marker='x', color='red', label='Received Symbols (With Filter)')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.title('QAM Modulation and Demodulation with Raised Cosine Filter')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()