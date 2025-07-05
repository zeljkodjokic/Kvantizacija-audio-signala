
# ...existing code...


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram
import scipy.stats
import io
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
from utils.noise import (
    add_white_noise, add_gaussian_noise, add_pink_noise, add_uniform_noise, add_impulse_noise, add_brownian_noise,
    add_poisson_noise, add_exponential_noise, add_rayleigh_noise, add_laplace_noise, add_gamma_noise, add_triangular_noise
)

st.title('Kvantizacija audio signala')

uploaded_files = st.file_uploader('Izaberi jedan ili više audio fajlova (WAV, MP3)', type=['wav', 'mp3'], accept_multiple_files=True)
num_bits = st.slider('Broj bitova kvantizacije', 2, 16, 8)
show_spectrogram = st.checkbox('Prikaži spektrogram', value=True)

for uploaded_file in uploaded_files:
    st.header(f'Fajl: {uploaded_file.name}')
    # Učitavanje audio fajla
    if uploaded_file.type == 'audio/wav':
        data, fs = sf.read(uploaded_file)
    else:
        st.warning('Za sada je podržan samo WAV format. MP3 podrška može biti dodata.')
        continue
    if data.ndim > 1:
        data = data[:, 0]  # samo prvi kanal
    st.audio(uploaded_file, format='audio/wav')

    # --- Dodavanje šuma ---
    st.subheader('Dodaj šum')
    noise_type = st.selectbox(
        'Tip šuma',
        [
            'Nema',
            'Beli šum (SNR)',
            'Gausovski šum (STD)',
            'Ružičasti šum (SNR)',
            'Uniformni šum (SNR)',
            'Impulsni šum',
            'Brownovski šum (SNR)',
            'Poissonov šum',
            'Eksponencijalni šum',
            'Rayleigh-ov šum',
            'Laplasov šum',
            'Gamma šum',
            'Trokutasti šum',
            # Dodaj još sumova ovde ako ih budeš implementirao
        ],
        key=f'noise_type_{uploaded_file.name}'
    )
    noisy_data = data.copy()
    added_noise = None
    if noise_type == 'Beli šum (SNR)':
        snr_db = st.slider('SNR za beli šum (dB)', 0, 40, 20, key=f'snr_{uploaded_file.name}')
        noisy_data, added_noise = add_white_noise(data, snr_db)
    elif noise_type == 'Gausovski šum (STD)':
        std = st.slider('Standardna devijacija gausovskog šuma', 0, 1, 0.1, step=0.01, key=f'std_{uploaded_file.name}')
        noisy_data, added_noise = add_gaussian_noise(data, std)
    elif noise_type == 'Ružičasti šum (SNR)':
        snr_db = st.slider('SNR za ružičasti šum (dB)', 0, 40, 20, key=f'snr_pink_{uploaded_file.name}')
        noisy_data, added_noise = add_pink_noise(data, snr_db)
    elif noise_type == 'Uniformni šum (SNR)':
        snr_db = st.slider('SNR za uniformni šum (dB)', 0, 40, 20, key=f'snr_uniform_{uploaded_file.name}')
        noisy_data, added_noise = add_uniform_noise(data, snr_db)
    elif noise_type == 'Impulsni šum':
        prob = st.slider('Verovatnoća impulsa', 0.0, 0.1, 0.01, step=0.001, key=f'prob_impulse_{uploaded_file.name}')
        amplitude = st.slider('Amplituda impulsa', 0.0, 2.0, 1.0, step=0.01, key=f'amp_impulse_{uploaded_file.name}')
        noisy_data, added_noise = add_impulse_noise(data, prob=prob, amplitude=amplitude)
    elif noise_type == 'Brownovski šum (SNR)':
        snr_db = st.slider('SNR za brownovski šum (dB)', 0, 40, 20, key=f'snr_brown_{uploaded_file.name}')
        noisy_data, added_noise = add_brownian_noise(data, snr_db)
    elif noise_type == 'Poissonov šum':
        lam = st.slider('Lambda (intenzitet) za Poissonov šum', 0.1, 10.0, 1.0, step=0.1, key=f'lam_poisson_{uploaded_file.name}')
        noisy_data, added_noise = add_poisson_noise(data, lam=lam)
    elif noise_type == 'Eksponencijalni šum':
        scale = st.slider('Scale (srednja vrednost) za eksponencijalni šum', 0.01, 2.0, 0.1, step=0.01, key=f'scale_exp_{uploaded_file.name}')
        noisy_data, added_noise = add_exponential_noise(data, scale=scale)
    elif noise_type == 'Rayleigh-ov šum':
        scale = st.slider('Scale za Rayleigh-ov šum', 0.01, 2.0, 0.1, step=0.01, key=f'scale_rayleigh_{uploaded_file.name}')
        noisy_data, added_noise = add_rayleigh_noise(data, scale=scale)
    elif noise_type == 'Laplasov šum':
        scale = st.slider('Scale (std) za Laplasov šum', 0.01, 2.0, 0.1, step=0.01, key=f'scale_laplace_{uploaded_file.name}')
        noisy_data, added_noise = add_laplace_noise(data, scale=scale)
    elif noise_type == 'Gamma šum':
        shape = st.slider('Shape za Gamma šum', 0.1, 10.0, 2.0, step=0.1, key=f'shape_gamma_{uploaded_file.name}')
        scale = st.slider('Scale za Gamma šum', 0.01, 2.0, 0.1, step=0.01, key=f'scale_gamma_{uploaded_file.name}')
        noisy_data, added_noise = add_gamma_noise(data, shape=shape, scale=scale)
    elif noise_type == 'Trokutasti šum':
        left = st.slider('Leva granica za trokutasti šum', -2.0, 0.0, -1.0, step=0.01, key=f'left_tri_{uploaded_file.name}')
        mode = st.slider('Mod za trokutasti šum', -1.0, 1.0, 0.0, step=0.01, key=f'mode_tri_{uploaded_file.name}')
        right = st.slider('Desna granica za trokutasti šum', 0.0, 2.0, 1.0, step=0.01, key=f'right_tri_{uploaded_file.name}')
        noisy_data, added_noise = add_triangular_noise(data, left=left, mode=mode, right=right)
    # --- Analiza šuma ---
    if noise_type != 'Nema' and added_noise is not None:
        st.subheader('Analiza šuma')
        fig_noise, ax_noise = plt.subplots(figsize=(10, 2))
        ax_noise.plot(added_noise[:1000], label='Šum (prvih 1000 uzoraka)')
        ax_noise.set_title('Šum - vremenski prikaz')
        ax_noise.legend()
        st.pyplot(fig_noise)

        fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
        ax_hist.hist(added_noise, bins=100, density=True, alpha=0.7)
        ax_hist.set_title('Histogram šuma')
        st.pyplot(fig_hist)

        # Prikaz spektra šuma
        from scipy.fft import rfft, rfftfreq
        N = len(added_noise)
        yf = np.abs(rfft(added_noise))
        xf = rfftfreq(N, 1/fs)
        fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
        ax_spec.plot(xf, 20 * np.log10(yf + 1e-12))
        ax_spec.set_title('Spektar šuma (dB)')
        ax_spec.set_xlabel('Frekvencija [Hz]')
        ax_spec.set_ylabel('Amplituda [dB]')
        st.pyplot(fig_spec)

        # Dodatne analize šuma
        st.markdown('**Statističke karakteristike šuma:**')
        st.write(f"Srednja vrednost: {np.mean(added_noise):.4e}")
        st.write(f"Standardna devijacija: {np.std(added_noise):.4e}")
        st.write(f"Skewness (asimetrija): {scipy.stats.skew(added_noise):.4f}")
        st.write(f"Kurtosis (zaravnjenost): {scipy.stats.kurtosis(added_noise):.4f}")

        # Minimum, maksimum, medijana, energija
        st.write(f"Minimum: {np.min(added_noise):.4e}")
        st.write(f"Maksimum: {np.max(added_noise):.4e}")
        st.write(f"Medijana: {np.median(added_noise):.4e}")
        st.write(f"Energija (suma kvadrata): {np.sum(added_noise**2):.4e}")

        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(added_noise)))[0]
        zcr = len(zero_crossings) / len(added_noise)
        st.write(f"Zero-crossing rate: {zcr:.4f}")

        # Percentili
        for p in [1, 5, 25, 50, 75, 95, 99]:
            st.write(f"{p}. percentil: {np.percentile(added_noise, p):.4e}")

        # Autokorelacija šuma (prvih 1000 uzoraka)
        st.markdown('**Autokorelacija šuma (prvih 1000 uzoraka):**')
        from statsmodels.tsa.stattools import acf
        acorr = acf(added_noise[:1000], nlags=40, fft=True)
        fig_acf, ax_acf = plt.subplots(figsize=(6, 2))
        ax_acf.stem(range(len(acorr)), acorr, use_line_collection=True)
        ax_acf.set_title('Autokorelacija šuma')
        ax_acf.set_xlabel('Kašnjenje')
        ax_acf.set_ylabel('ACF')
        st.pyplot(fig_acf)

        # DFT fazni spektar
        from scipy.fft import rfft, rfftfreq
        N = len(added_noise)
        yf = rfft(added_noise)
        phase = np.angle(yf)
        xf = rfftfreq(N, 1/fs)
        fig_phase, ax_phase = plt.subplots(figsize=(10, 2))
        ax_phase.plot(xf, phase)
        ax_phase.set_title('Fazni spektar šuma')
        ax_phase.set_xlabel('Frekvencija [Hz]')
        ax_phase.set_ylabel('Faza [rad]')
        st.pyplot(fig_phase)

        # Napredne analize šuma
        st.markdown('**Dodatne napredne analize šuma:**')
        # RMS vrednost
        rms = np.sqrt(np.mean(added_noise**2))
        st.write(f"RMS vrednost: {rms:.4e}")

        # Entropija šuma (Shannon entropy)
        hist, bin_edges = np.histogram(added_noise, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        st.write(f"Shannon entropija: {entropy:.4f} bita")

        # Signal-to-Noise Ratio (SNR) u dB (provera)
        snr_db = 10 * np.log10(np.sum(noisy_data**2) / np.sum(added_noise**2))
        st.write(f"SNR (noisy/added_noise): {snr_db:.2f} dB")

        # Ljung-Box test na autokorelaciju (test nasumičnosti)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(added_noise[:1000], lags=[10], return_df=True)
            st.write(f"Ljung-Box p-vrednost (lag 10): {lb_test['lb_pvalue'].values[0]:.4f}")
        except Exception as e:
            st.write(f"Ljung-Box test nije uspeo: {e}")

        # Detrended Fluctuation Analysis (DFA) - za procenu fraktalne dimenzije (ako je dostupan paket)
        try:
            import nolds
            alpha = nolds.dfa(added_noise[:10000])
            st.write(f"DFA (fraktalna dimenzija): {alpha:.4f}")
        except ImportError:
            st.write("nolds nije instaliran (DFA analiza nije dostupna)")
        except Exception as e:
            st.write(f"DFA analiza nije uspela: {e}")

    # Kvantizacija
    min_val, max_val = -1, 1
    levels = 2 ** num_bits
    step = (max_val - min_val) / (levels - 1)
    quantized = np.round((noisy_data - min_val) / step) * step + min_val
    quantized = np.clip(quantized, min_val, max_val)

    # Prikaz signala
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data[:1000], label='Original')
    if noise_type != 'Nema':
        ax.plot(noisy_data[:1000], label='Sa šumom', alpha=0.7)
    ax.step(range(1000), quantized[:1000], label='Kvantizovan', where='mid')
    ax.set_title('Signal (prvih 1000 uzoraka)')
    ax.legend()
    st.pyplot(fig)

    # SNR
    noise = noisy_data - quantized
    snr = 10 * np.log10(np.sum(noisy_data**2) / np.sum(noise**2))
    st.write(f'SNR nakon kvantizacije: {snr:.2f} dB')
    if noise_type != 'Nema' and added_noise is not None:
        snr_noise = 10 * np.log10(np.sum(data**2) / np.sum(added_noise**2))
        st.write(f'SNR original vs. šum: {snr_noise:.2f} dB')

    # Spektrogram
    if show_spectrogram:
        f, t, Sxx = spectrogram(data, fs)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax2.set_ylabel('Frekvencija [Hz]')
        ax2.set_xlabel('Vreme [s]')
        ax2.set_title('Spektrogram originalnog signala')
        st.pyplot(fig2)
        if noise_type != 'Nema':
            f, t, Sxx = spectrogram(noisy_data, fs)
            fign, axn = plt.subplots(figsize=(10, 3))
            axn.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            axn.set_ylabel('Frekvencija [Hz]')
            axn.set_xlabel('Vreme [s]')
            axn.set_title('Spektrogram signala sa šumom')
            st.pyplot(fign)
        f, t, Sxx = spectrogram(quantized, fs)
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax3.set_ylabel('Frekvencija [Hz]')
        ax3.set_xlabel('Vreme [s]')
        ax3.set_title('Spektrogram kvantizovanog signala')
        st.pyplot(fig3)
