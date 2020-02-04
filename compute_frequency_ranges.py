# Imports
import math
import numpy as np


# Greenwood function: F = A(10^(ax)-k) (x = 0 at apex, 1 at base)
# The constants below are used to mimic the human cochlea, as described in Greenwood, D. D. (1990). A cochlear
# frequency‐position function for several species—29 years later. The Journal of the Acoustical Society of America,
# 2592-2605.
A = 165.4
a = 2.1
k = 1


# Parameters
x_low = 0.133489 # F = 150 Hz
x_high = 0.806376 # F = 7999 Hz
x_range = x_high - x_low


# Computes the frequency ranges for a number of channels in a cochlear implant between 150 Hz and 8000 Hz, using the
# Greenwood function.
# nr_channels: the number of channels for which the frequency ranges are computed
def compute_frequency_ranges(nr_channels):

    # Step size between channels
    step_size = x_range / nr_channels

    # Determine range frequencies
    freqs = np.zeros(nr_channels+1)
    for i in range(0, nr_channels+1):
        x = x_low + i * step_size
        freqs[i] = int(A * (math.pow(10, a * x) - k))

    # Create frequency range tuples
    freq_ranges = np.zeros(nr_channels, dtype='2i')
    for i in range(0, nr_channels):
        freq_ranges[i] = (freqs[i], freqs[i+1])

    # Save to txt and npy file
    np.save("frequency_ranges.npy", freq_ranges)
    np.savetxt("frequency_ranges.txt", freq_ranges, fmt='%s')

    return freq_ranges