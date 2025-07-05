import numpy as np
import matplotlib.pyplot as plt

# Parametri signala
duration = 1.0  # trajanje u sekundama
fs = 8000       # frekvencija uzorkovanja (Hz)
f = 440         # frekvencija sinusoide (Hz)
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
signal = 0.8 * np.sin(2 * np.pi * f * t)

# Kvantizacija
num_bits = 3  # broj bitova kvantizacije
levels = 2 ** num_bits
min_val, max_val = -1, 1
step = (max_val - min_val) / (levels - 1)
quantized_signal = np.round((signal - min_val) / step) * step + min_val
quantized_signal = np.clip(quantized_signal, min_val, max_val)

# SNR
noise = signal - quantized_signal
snr = 10 * np.log10(np.sum(signal**2) / np.sum(noise**2))
print(f'SNR: {snr:.2f} dB')

# Prikaz signala
plt.figure(figsize=(10, 5))
plt.plot(t[:200], signal[:200], label='Originalni signal')
plt.step(t[:200], quantized_signal[:200], label='Kvantizovani signal', where='mid')
plt.xlabel('Vreme [s]')
plt.ylabel('Amplituda')
plt.title('Kvantizacija sinusoide')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
