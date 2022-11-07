import subprocess

import numpy as np
from numpy import ndarray

import Constants


def pca(data: list[list]) -> ndarray:
    data_path: str = Constants.pca_data_path
    save_before_apply_pca(data, data_path)
    commands: list[str] = [*Constants.pca_loader, data_path, f"{Constants.num_components}"]

    loaded_audio = subprocess.check_output(commands)
    loaded_audio: str = loaded_audio.decode("utf-8")
    return pca_post_process(loaded_audio)


def pca_post_process(data: str) -> ndarray:
    # data == 2d array
    tmp = []
    for d in data.splitlines():
        tmp.append([float(dd) for dd in d.split(",")])
    result: ndarray = np.array(tmp)

    return result


def save_before_apply_pca(data: list[list], data_path: str):
    with open(data_path, "w+", encoding="utf-8") as f:
        # line = ""
        for r in data:
            line = ",".join(str(d) for d in r)
            f.writelines(line + "\n")
    print(f"[DONE] data write in {data_path}")
