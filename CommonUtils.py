import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA

import Constants


def save_prep_data(data: list[list], data_path: str) -> None:
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


# def save_applied_pca_data(data: list, path: str) -> None:
#     with open(path, "w+", encoding="utf-8") as f:


def save_eigen_values(data: list, path: str) -> None:
    with open(path, "w+", encoding='utf-8') as f:
        f.writelines((",".join([str(d) for d in data])).replace("[", "").replace("]", ""))
    print(f"eigen value 저장 완료 - {path}")


def save_mean_values(data: list, path: str) -> None:
    with open(path, "w+", encoding='utf-8') as f:
        f.writelines((",".join([str(d) for d in data])).replace("[", "").replace("]", ""))

    print(f"mean value 저장 완료 - {path}")


def apply_pca(data: list, mean_path: str, eigen_path) -> tuple[ndarray, ndarray, ndarray]:
    pca = PCA(Constants.num_components)
    data = np.asarray(data, dtype=float)

    result = pca.fit_transform(data)

    mean_values = pca.mean_

    save_mean_values(mean_values.tolist(), mean_path)
    eigen_vector = pca.components_  # eigen values
    save_eigen_values(eigen_vector.tolist(), eigen_path)
    return result, mean_values, eigen_vector


def test_pca(data: ndarray, mean: ndarray, eigen: ndarray) -> ndarray:
    # pca 결과 계산
    # (file count, 50), (50, file count)
    # (1000, file count), (50, 10_000)
    # eigen=(50x10K), data-mean=(10K, file_count)
    sub_result = data - mean
    result = np.dot(sub_result, eigen.T)
    return result
