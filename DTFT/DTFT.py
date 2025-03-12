import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from scipy.signal.windows import hann, flattop

f1 = 400       
f2 = 400.25
f3 = 399.75    
A = 4          
fs = 600       
N = 3000       

k = np.arange(N)

x1 = A * np.sin(2 * np.pi * f1/fs * k)
x2 = A * np.sin(2 * np.pi * f2/fs * k)
x3 = A * np.sin(2 * np.pi * f3/fs * k)

x = x1 + x2 + x3

w_rect = np.ones(N) 
w_hann = hann(N, sym=False) 
w_flattop = flattop(N, sym=False)

plt.figure(figsize=(10, 4))
plt.plot(k, w_rect, 'C0o-', markersize=3, label='Rectangular')
plt.plot(k, w_hann, 'C1o-', markersize=3, label='Hann')
plt.plot(k, w_flattop, 'C2o-', markersize=3, label='Flattop')
plt.xlabel('k')
plt.ylabel('Window value')
plt.xlim(0, N)
plt.legend()
plt.title('Window Functions')
plt.grid(True)
plt.show()

def fft2db(X):
    """
    Returns the FFT level in dB normalized for sine signal amplitudes.
    Multiplication by 2/N normalizes the amplitude independent of N.
    Note: The DC bin and (if present) the fs/2 bin occur only once.
    """
    N_fft = X.size
    Xtmp = 2 / N_fft * X.copy()  
    Xtmp[0] *= 0.5             
    if N_fft % 2 == 0:          
        Xtmp[N_fft//2] /= 2
    return 20 * np.log10(np.abs(Xtmp))

df = fs / N
f_vec = np.arange(N) * df

X_rect = fft(x * w_rect)
X_hann = fft(x * w_hann)
X_flattop = fft(x * w_flattop)

f_min = 190
f_max = 210
half = N // 2 

plt.figure(figsize=(16, 10))

plt.subplot(3, 1, 1)
plt.plot(f_vec[:half], fft2db(X_rect)[:half], 'C0o-', markersize=3, label='Rectangular')
plt.xlim(f_min, f_max)
plt.ylim(-50, 0)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Normalized FFT Spectrum (Rectangular Window)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(f_vec[:half], fft2db(X_hann)[:half], 'C1o-', markersize=3, label='Hann')
plt.xlim(f_min, f_max)
plt.ylim(-50, 0)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Normalized FFT Spectrum (Hann Window)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(f_vec[:half], fft2db(X_flattop)[:half], 'C2o-', markersize=3, label='Flattop')
plt.xlim(f_min, f_max)
plt.ylim(-50, 0)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Normalized FFT Spectrum (Flattop Window)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

def winDTFTdB(w):
    """
    Returns the quasi-DTFT of a window 'w' in dB, normalized to its mainlobe maximum.
    Zero-padding is used to increase the frequency resolution.
    """
    N_w = w.size
    Nz = 100 * N_w   
    W = np.zeros(Nz)
    W[:N_w] = w      
    W = np.abs(fftshift(fft(W)))  
    W = W / np.max(W)  
    W_db = 20 * np.log10(W)
    Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi  
    return Omega, W_db

Omega_rect, W_rect_db = winDTFTdB(w_rect)
Omega_hann, W_hann_db = winDTFTdB(w_hann)
Omega_flattop, W_flattop_db = winDTFTdB(w_flattop)

plt.figure(figsize=(16, 10))

plt.subplot(2, 1, 1)
plt.plot(Omega_rect, W_rect_db, label='Rectangular')
plt.plot(Omega_hann, W_hann_db, label='Hann')
plt.plot(Omega_flattop, W_flattop_db, label='Flattop')
plt.xlabel('Digital Frequency Ω (rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.title('Window DTFT Spectra (Full Range)')
plt.xlim(-np.pi, np.pi)
plt.ylim(-120, 10)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(Omega_rect, W_rect_db, label='Rectangular')
plt.plot(Omega_hann, W_hann_db, label='Hann')
plt.plot(Omega_flattop, W_flattop_db, label='Flattop')
plt.xlabel('Digital Frequency Ω (rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.title('Window DTFT Spectra (Mainlobe Zoom)')
plt.xlim(-np.pi/100, np.pi/100)
plt.ylim(-120, 10)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
