import numpy as np
from tqdm import tqdm

from Constants import vibe_offset, vibe_cut_time, vibe_sr


# def sliding_window(arr, win_size=88200, step_size=44100, copy=False):
#     result: list = []
#
#     idx: int = 0
#     while len(arr) > idx + win_size:
#         tmp_window: list = arr[idx:idx + win_size]
#         result.append(np.array(tmp_window))
#         idx += step_size


def load_vib_file(file: str) -> list[float]:
    vib_data: list[float] = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            vib_data.append(float(line.split("\t")[-1]))
    return vib_data


def extract_vib_features(file: str) -> list:
    # extracted_features = []

    # 데이터 읽기
    vib_data = np.array(load_vib_file(file))
    # 8초로 자르기
    vib_data = vib_data[vibe_sr * vibe_offset: vibe_sr * (vibe_cut_time + vibe_offset)]

    # for signal in sliding_window(vib_data):
    fft = np.fft.fft(vib_data)
    # extracted_features.append()

    return np.abs(fft).tolist()


def get_features_from_files_vib(list_files: list[list]):
    list_features = []
    for idx, file in tqdm(enumerate(list_files), total=len(list_files)):
        list_features.append(extract_vib_features(file))
    return list_features
