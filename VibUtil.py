import os

import numpy as np
import re
from parmap import parmap

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
            vib_data.append(float(re.split("\s", line.strip())[-1]))
    return vib_data


def extract_vib_features(file: str) -> list:
    # extracted_features = []

    # 데이터 읽기
    vib_data = np.asarray(load_vib_file(file), dtype=float)
    vib_data = vib_data[int(vibe_sr * vibe_offset): int(vibe_sr * (vibe_cut_time + vibe_offset))]

    # print(f"file -- {file} / load_length :: {len(vib_data)}")

    if len(vib_data) / vibe_sr < vibe_cut_time:
        return []
    # 8초로 자르기

    # for signal in sliding_window(vib_data):
    fft = np.fft.fft(vib_data)
    # extracted_features.append()
    a = 1
    return np.abs(fft).tolist()


def parallel_get_features_from_files_vib(list_files: list[str]) -> list[list]:
    result = parmap.map(extract_vib_features, list_files, pm_pbar=True, pm_processes=os.cpu_count())
    return list(filter(lambda arr: len(arr) != 0, result))
