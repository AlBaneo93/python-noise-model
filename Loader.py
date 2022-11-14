import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy import ndarray

import Constants
from Logger import logger
from model_runner import Duration


# def load_parallel(file_list: list[str]):
#     p = Pool(processes=mp.cpu_count())
#
#     p.map(loader, file_list)


@Duration
def load_audio(file_name: str) -> ndarray:
    output_name: str = make_output_name(file_name)

    logger.info("Run Audio Load task")
    commands: list[str] = [*Constants.audio_loader, file_name, output_name]

    loaded_audio = subprocess.check_output(commands)
    loaded_audio = loaded_audio.decode("utf-8")
    logger.info("Done Audio Load task")
    return post_process(loaded_audio)


@Duration
def post_process(result: str) -> ndarray:
    result_list = result.replace("\b", "") \
        .replace(" ", "") \
        .replace("[", "") \
        .replace("]", "") \
        .replace("\\n", "") \
        .replace("\n", "") \
        .replace("\"", "") \
        .split(",")
    return np.array(list(map(float, result_list)))


def make_output_name(input_name: str) -> str:
    arr = input_name.split("/")
    return f"{Constants.resample_result_path}/resampled_{arr[-1]}"


@Duration
def clear_before_resample():
    shutil.rmtree(Constants.resample_result_path)
    Path(Constants.resample_result_path).mkdir(Constants.resample_result_path, parents=True, exist_ok=True)
