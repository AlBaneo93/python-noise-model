import os

import librosa
import numpy as np
from numpy import ndarray
from parmap import parmap

from Constants import noise_sr, n_ffts, n_mels, n_step, noise_offset, noise_cut_time
from Loader import load_audio
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
def reshape_features(features: ndarray) -> ndarray:
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

    loaded_data: ndarray = load_audio(file)
    if len(loaded_data) / noise_sr < noise_cut_time:
        return []
    loaded_data = loaded_data[noise_sr * noise_offset:noise_sr * (noise_cut_time + noise_offset)]
    # signal_divided: ndarray = sliding_window(loaded_data)

    # Extract Mel-spectrogram from each divided signal
    mel_fbank = get_mel(sr=noise_sr, n_ffts=n_ffts, n_mels=n_mels)

    # for signal in signal_divided:
    # NOTE : shape : (1025, 22)
    data_mel: ndarray = librosa.stft(y=loaded_data, n_fft=n_ffts, hop_length=n_step)

    data_mel = np.dot(np.abs(data_mel.T), mel_fbank.T)
    data_mel = power_to_decibel(data_mel).T
    # print(n_mels)
    # # extracted_features.append(data_mel.T)
    # # TODO : 1D 형태가 되는지 확인하기
    # NOTE : shape - (16,22) (reshape 적용 전 shape)
    # data_mel = reshape_features(data_mel)
    return my_reshape(data_mel)
    # return extracted_features # 3d array


def my_reshape(arr: ndarray) -> list:
    """

    :param arr: 2D ndarray
    :return: 1D list
    """
    result = []
    w, d = arr.shape
    return np.reshape(arr, w * d).tolist()


# Extract features from a list of files
# def get_features_from_files(list_files) -> list[list]:
#     list_features = []
#
#     pool = Pool()
#
#     before_t_result = 0
#     for idx, file in tqdm(enumerate(list_files), total=len(list_files)):
#         # list_features = [*list_features, *extract_features(file)]
#         t_result = extract_features(file)
#         if len(t_result) == 0:
#             continue
#
#         list_features.append(t_result)
#     # NOTE : 2D Array shape
#     return list_features


def parallel_get_features_from_files(list_files: list[str]) -> list[list]:
    result = parmap.map(extract_features, list_files, pm_pbar=True, pm_processes=os.cpu_count())
    return list(filter(lambda arr: len(arr) != 0, result))
