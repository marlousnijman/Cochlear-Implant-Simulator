# Based on:
#Fu, Z. (2019, May 5). YannyLaurel-with-CochlearImplant-Simulation. Retrieved from Github:
# https://github.com/fuzhenfz/YannyLaurel-with-CochlearImplant-Simulation

# Imports
from scipy.signal import hilbert, lfilter, firwin
import numpy as np


# Creates sine waves that correspond to the average of the frequency ranges of each channel.
# freq: array of frequency ranges for the number of channels
# nsample: the length of the data that will be used
# rate: the sampling rate of the data
def create_sin_waves(freq_ranges, nsample, rate):
    nfreq = len(freq_ranges)
    t = np.arange(nsample) / rate
    sin_waves = np.zeros((nfreq, len(t)))

    for i in range(0, nfreq):
        fc = (freq_ranges[i][0] + freq_ranges[i][1]) / 2
        sin_waves[i] = np.sin(2 * np.pi * fc * t + (i) * 2 * np.pi / 10)

    return sin_waves


# Cochlear implant simulator, based on continuous interleaved sampling (CIS).
# data: data that will be converted
# sin_waves: sine waves that are needed to simulate the cochlear implant
# rate: sampling rate of the data
# freq_ranges: frequency ranges corresponding to the channels of the to be simulated cochlear implant
# numtaps: length of the firwin filter
def cis(data, rate, freq_ranges, numtaps=1025):
    nfreq = len(freq_ranges)
    nsample = len (data)
    data = data / 32767 # from 16 to 32 bit representation
    sim_data = np.zeros(nsample)
    sin_waves = create_sin_waves(freq_ranges, nsample, rate)

    for i in range(0, nfreq):
        b = firwin(numtaps, np.asarray([freq_ranges[i][0], freq_ranges[i][1]]), fs=rate, pass_zero=False)
        signal_filtered = lfilter(b, 1, data)
        analytic_signal = hilbert(signal_filtered)
        amplitude_envelope = np.abs(analytic_signal)
        sim_data = sim_data + amplitude_envelope * sin_waves[i]

    sim_data = sim_data / (max(abs(sim_data))) * 32767 # from 32 to 16 bit representation

    return sim_data