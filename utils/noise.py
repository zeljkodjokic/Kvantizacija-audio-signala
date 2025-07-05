def add_poisson_noise(signal, lam=1.0):
    """
    Dodaje Poissonov šum na signal (koristi se za simulaciju brojačkih procesa, npr. fotoni u slici).
    lam: lambda parametar Poissonove raspodele
    """
    # Poisson noise se obično koristi za nenormalizovane signale (npr. slike), ali može i za audio
    scaled = signal - np.min(signal)
    scaled = scaled / np.max(scaled) * lam
    noise = np.random.poisson(scaled) - scaled
    return signal + noise, noise

def add_exponential_noise(signal, scale=1.0):
    """
    Dodaje eksponencijalni šum na signal.
    scale: srednja vrednost eksponencijalne raspodele
    """
    noise = np.random.exponential(scale, size=signal.shape) - scale
    return signal + noise, noise

def add_rayleigh_noise(signal, scale=1.0):
    """
    Dodaje Rayleigh-ov šum na signal.
    scale: parametar Rayleigh raspodele
    """
    noise = np.random.rayleigh(scale, size=signal.shape) - scale * np.sqrt(np.pi / 2)
    return signal + noise, noise

def add_laplace_noise(signal, scale=1.0):
    """
    Dodaje Laplasov (double exponential) šum na signal.
    scale: standardna devijacija šuma
    """
    noise = np.random.laplace(0, scale, size=signal.shape)
    return signal + noise, noise

def add_gamma_noise(signal, shape=2.0, scale=1.0):
    """
    Dodaje Gamma šum na signal.
    shape: oblik raspodele
    scale: skala raspodele
    """
    noise = np.random.gamma(shape, scale, size=signal.shape) - shape * scale
    return signal + noise, noise

def add_triangular_noise(signal, left=-1.0, mode=0.0, right=1.0):
    """
    Dodaje trokutasti (triangular) šum na signal.
    left, mode, right: parametri raspodele
    """
    noise = np.random.triangular(left, mode, right, size=signal.shape)
    mean = (left + mode + right) / 3
    noise = noise - mean
    return signal + noise, noise
def add_uniform_noise(signal, snr_db):
    """
    Dodaje uniformni šum na signal za zadati SNR u dB.
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    # Uniformni šum u opsegu [-a, a], varijansa = a^2/3 => a = sqrt(3*noise_power)
    a = np.sqrt(3 * noise_power)
    noise = np.random.uniform(-a, a, size=signal.shape)
    return signal + noise, noise

def add_impulse_noise(signal, prob=0.01, amplitude=1.0):
    """
    Dodaje impulsni (salt-and-pepper) šum na signal.
    prob: verovatnoća impulsa po uzorku
    amplitude: vrednost impulsa
    """
    noise = np.zeros_like(signal)
    mask = np.random.rand(*signal.shape) < prob
    noise[mask] = amplitude * np.random.choice([-1, 1], size=np.sum(mask))
    return signal + noise, noise

def add_brownian_noise(signal, snr_db):
    """
    Dodaje brownovski (Brownian/red) šum na signal za zadati SNR u dB.
    """
    N = len(signal)
    # Brownian noise: integral belog šuma
    white = np.random.randn(N)
    brown = np.cumsum(white)
    brown = brown - np.mean(brown)
    brown = brown / np.std(brown)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    brown_noise = brown * np.sqrt(noise_power)
    return signal + brown_noise, brown_noise


import numpy as np

def add_white_noise(signal, snr_db):
    """
    Dodaje beli šum na signal za zadati SNR u dB.
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise, noise


def add_gaussian_noise(signal, std):
    """
    Dodaje gausovski šum sa zadatim standardnim odstupanjem.
    """
    noise = np.random.normal(0, std, size=signal.shape)
    return signal + noise, noise

def add_pink_noise(signal, snr_db):
    """
    Dodaje ružičasti (pink) šum na signal za zadati SNR u dB.
    """
    # Pink noise: 1/f karakteristika
    N = len(signal)
    uneven = N % 2
    X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 da izbegnemo deljenje sa nulom
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    y = y / np.std(y)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    pink_noise = y * np.sqrt(noise_power)
    return signal + pink_noise, pink_noise
