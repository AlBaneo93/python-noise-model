from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#### Compute Mel based on Slaney formula

# Convert Hz to Mels
def hz_to_mel(freq):
    freq = np.asanyarray(freq)

    # The linear part
    f_sp = 200.0 / 3
    mels = freq / f_sp

    # Log-scale part
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = np.log(6.4) / 27.0

    if freq.ndim:  # vectorize the given array
        log_t = freq >= min_log_hz
        mels[log_t] = min_log_mel + np.log(freq[log_t] / min_log_hz) / log_step
    elif freq >= min_log_hz:
        mels = min_log_mel + np.log(freq / min_log_hz) / log_step
    return mels

# Convert mel bin numbers to freqies
def mel_to_hz(mels):
    mels = np.asanyarray(mels)

    # The linear scale
    f_sp = 200.0 / 3
    freqs = f_sp * mels

    # The nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = np.log(6.4) / 27.0
    
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(log_step * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(log_step * (mels - min_log_mel))
    return freqs

def mel_filterbank(f_max=11025.0, n_mels=64):
    mels = np.linspace(0, hz_to_mel(f_max), n_mels)
    return mel_to_hz(mels)

def get_mel(sr, n_ffts, n_mels=64):
    # Initialize the weights
    f_max = float(sr) / 2
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_ffts // 2)), dtype=np.float32)

    # Center freqs of FFT bins
    fft_freq = np.fft.rfftfreq(n=n_ffts, d=1.0 / sr)

    # Center freqs of mel bands
    mel_freq = mel_filterbank(f_max, n_mels+2)

    freq_diff = np.diff(mel_freq)
    ramps = np.subtract.outer(mel_freq, fft_freq)

    for i in range(n_mels):
        lower = -ramps[i] / freq_diff[i]
        upper = ramps[i+2] / freq_diff[i+1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Normalization to be approx constant energy per channel
    enorm = 2.0 / (mel_freq[2:n_mels+2] - mel_freq[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights