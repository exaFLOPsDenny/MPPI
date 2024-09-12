import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Generate some noisy data
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

# Apply Savitzky-Golay filter
window_length = 11  # Choose odd window length
poly_order = 3      # Polynomial order
y_smooth = savgol_filter(y, window_length, poly_order)

# Plot the results
plt.plot(x, y, label='Noisy Signal')
plt.plot(x, y_smooth, label='Smoothed Signal (Savitzky-Golay)', color='red')
plt.legend()
plt.show()
