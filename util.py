import numpy as np
from numpy import ndarray
from tqdm import tqdm
import librosa

import Constants
from Constants import sr, n_ffts, n_mels, n_step, offset, cut_time
from Loader import loader
from melspec import get_mel


def sliding_window(arr: ndarray, sample_rate: int = 22050) -> ndarray:
    step_size = sample_rate * 4
    win_size = sample_rate * 8
    idx = 0
    result: list[list] = []
    while arr.size > idx + win_size:
        tmp_window = arr[idx: idx + win_size]
        result.append(tmp_window)
        idx += step_size

    return np.array(result)


# Reshape the extracted features so to be 1-dim arrays
def reshape_features(features):
    if len(features) > 0:
        w, d = features[0].shape
        sz_features = len(features)
        reshaped = np.zeros((sz_features, (w * d)))
        for i in range(sz_features):
            reshaped[i] = np.reshape(features[i], w * d)
    else:
        reshaped = []
    return reshaped


# Convert power densities as decibel scale
def power_to_decibel(S, *, ref=1.0, amin=1e-10, top_db=80.0) -> ndarray:
    magnitude = np.abs(S)

    # ref_value = np.max(magnitude)
    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


# Convert sound amplitude as decibel scale
def amplitude_to_decibel(S, amin=1e-5, top_db=80.0):
    magnitude = np.abs(S)
    ref_value = np.max(magnitude)
    power = np.square(magnitude, out=magnitude)
    return power_to_decibel(power, ref=ref_value ** 2, amin=amin ** 2, top_db=top_db)


# Extract features from an .wav file
def extract_features(file) -> list:
    # extracted_features = []

    loaded_data: ndarray = loader(file)
    loaded_data = loaded_data[sr(0 + offset):sr * (cut_time + offset)]
    # signal_divided: ndarray = sliding_window(loaded_data)

    # Extract Mel-spectrogram from each divided signal
    mel_fbank = get_mel(sr=sr, n_ffts=n_ffts, n_mels=n_mels)

    # for signal in signal_divided:
    data_mel = librosa.stft(y=loaded_data, n_fft=n_ffts, hop_length=n_step)
    data_mel = np.dot(np.abs(data_mel.T), mel_fbank.T)
    data_mel = power_to_decibel(data_mel).T
    # print(n_mels)
    # # extracted_features.append(data_mel.T)
    # # TODO : 1D 형태가 되는지 확인하기
    return data_mel
    # return extracted_features # 3d array


# Extract features from a list of files
def get_features_from_files(list_files) -> list[list]:
    list_features = []
    for idx, file in tqdm(enumerate(list_files), total=len(list_files)):
        list_features.append(extract_features(file))
    # NOTE : 2D Array shape
    return list_features
