import subprocess

import numpy as np
from numpy import ndarray

from Constants import pca_loader, num_components


def pca(data: list[list], before_save_path: str, result_save_path: str) -> ndarray:
    """

    :param data: pca를 적용할 데이터
    :param before_save_path: pca 모듈이 부르게 될 데이터의 위치
    :param result_save_path: [for test] pca 모듈이 저장한 pca가 적용된 데이터의 위치
    :return: pca가 적용된 데이터
    """
    save_before_apply_pca(data, before_save_path)
    a = 1
    commands: list[str] = [*pca_loader, before_save_path, f"{num_components}"]

    loaded_audio = subprocess.check_output(commands)
    loaded_audio: str = loaded_audio.decode("utf-8")
    result: ndarray = pca_post_process(loaded_audio)
    save_after_pca(result, result_save_path)
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


def save_after_pca(data: ndarray, after_save_path: str) -> None:
    """

    :param data: 2D ndarray
    :return:
    """
    data = data.tolist()

    with open(after_save_path, "w+", encoding="utf-8") as f:
        for row in data:
            line = ",".join(str(d) for d in row)
            f.writelines(f"{line}\n")
    print(f"[DONE] Write after pca data in {after_save_path}")
