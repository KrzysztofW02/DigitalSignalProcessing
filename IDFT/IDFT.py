import numpy as np
import matplotlib.pyplot as plt

mu = np.array([10, 5, 6, 6, 2, 4, 3, 4, 5, 0, 0, 0, 0], dtype=complex)
N = len(mu)  # Number of samples

n = np.arange(N).reshape(N, 1)  
k = np.arange(N).reshape(1, N)  
W = np.exp(1j * 2 * np.pi * n * k / N)  # Fourier matrix

x = np.real((1 / N) * W @ mu)  

plt.stem(np.arange(N), x)  

for i in range(N):
    plt.text(i, x[i] + 0.05, f"{x[i]:.2f}", ha='center', fontsize=10)

plt.xlabel("n (samples)")  
plt.ylabel("x[n]")  
plt.title("Synthesized signal from IDFT")  
plt.grid()
plt.show()
