import subprocess

import numpy as np
from numpy import ndarray

from Constants import pca_loader, pca_data_path, num_components, after_pca_data_path


def pca(data: list[list]) -> ndarray:
    a = 1

    save_before_apply_pca(data, pca_data_path)
    commands: list[str] = [*pca_loader, pca_data_path, f"{num_components}"]

    loaded_audio = subprocess.check_output(commands)
    loaded_audio: str = loaded_audio.decode("utf-8")
    result: ndarray = pca_post_process(loaded_audio)
    save_after_pca(result)
    return result


def pca_post_process(data: str) -> ndarray:
    """

    :param data: PCA.jar 에서 가져온 PCA데이터
    :return: 후처리가 적용된 PCA 데이터
    """
    # data == 2d array
    tmp = []
    for d in data.splitlines():
        tmp.append([float(dd) for dd in d.split(",")])
    result: ndarray = np.array(tmp)

    return result


def save_before_apply_pca(data: list[list], data_path: str) -> None:
    """

    :param data: 전처리 된 데이터
    :param data_path: 전처리 된 데이터의 값이 저장될 경로
    """
    with open(data_path, "w+", encoding="utf-8") as f:
        # line = ""
        for r in data:
            line = ",".join(str(d) for d in r)
            f.writelines(line + "\n")
    print(f"[DONE] data write in {data_path}")


def save_after_pca(data: ndarray) -> None:
    """

    :param data: 2D ndarray
    :return:
    """
    data = data.tolist()

    with open(after_pca_data_path, "w+", encoding="utf-8") as f:
        for row in data:
            line = ",".join(str(d) for d in row)
            f.writelines(f"{line}\n")
    print(f"[DONE] Write after pca data in {after_pca_data_path}")
