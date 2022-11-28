# Data loading
import datetime
import glob
import os
import time

# Python libraries
import numpy as np
from libsvm.svmutil import svm_train, svm_save_model, svm_predict
from numpy import ndarray

from CommonUtils import save_prep_data, apply_pca, test_pca
from Constants import noise_path_train, noise_path_test, \
    svm_cost, svm_kernel, noise_model_save_path, \
    noise_train_prefix, noise_pc_value_path, noise_mean_value_path, test_noise_mean_value_path, test_noise_pc_value_path
# Sound processing
from util import parallel_get_features_from_files

train_mean = np.array([])
train_eigen = np.array([])


def get_train_dataset_noise() -> tuple[ndarray, ndarray]:
    print('[NOISE][LOAD] Extracting features from train dataset: {}'.format(noise_path_train))
    list_files_normal: list[str] = sorted(glob.glob(os.path.join(noise_path_train, 'normal_*.wav')))
    list_files_abnormal: list[str] = sorted(glob.glob(os.path.join(noise_path_train, 'anomal_*.wav')))
    list_files_reduce: list[str] = sorted(glob.glob(os.path.join(noise_path_train, 'reducer_*.wav')))

    if len([*list_files_normal, *list_files_abnormal, *list_files_reduce]) != 0:
        data_mel_normal = parallel_get_features_from_files(list_files_normal)
        data_mel_abnormal = parallel_get_features_from_files(list_files_abnormal)
        data_mel_reduce = parallel_get_features_from_files(list_files_reduce)

        print(f"[NOISE] 학습 데이터의 수 :: {len([*data_mel_normal, *data_mel_abnormal, *data_mel_reduce])}")
        kph_dataset = data_mel_normal + data_mel_abnormal + data_mel_reduce  # 2d array

        # Make labels
        kph_labels_normal = [0 for _ in data_mel_normal]
        kph_labels_abnormal = [1 for _ in data_mel_abnormal]
        kph_labels_reduce = [2 for _ in data_mel_reduce]
        kph_labels = np.array(kph_labels_normal + kph_labels_abnormal + kph_labels_reduce)

        # return kph_labels, np.array(kph_dataset)
        # reshape the train dataset
        # kph_dataset_reshaped = reshape_features(kph_dataset)
        # NOTE : 2D array여야 함
        #     kph_dataset_reshaped: ndarray = pca(kph_dataset,
        #                                         noise_before_pca_data_path,
        #                                         noise_after_pca_data_path,
        #                                         noise_pc_value_path,
        #                                         noise_mean_value_path,
        #                                         noise_pca_result_path)
        pca_result, mean, eigen = apply_pca(kph_dataset, noise_mean_value_path, noise_pc_value_path)
        global train_mean
        global train_eigen
        train_mean = np.asarray(mean)
        train_eigen = np.asarray(eigen)
        return kph_labels, pca_result

    return np.array([]), np.array([])


def get_test_dataset_noise() -> tuple[ndarray, ndarray]:
    # Test dataset
    print('[NOISE][LOAD] Extracting features from test dataset: {}'.format(noise_path_test))

    list_files_normal_test: list[str] = sorted(glob.glob(os.path.join(noise_path_test, 'Test_normal_*.wav')))
    list_files_abnormal_test: list[str] = sorted(glob.glob(os.path.join(noise_path_test, 'Test_anomal_*.wav')))
    list_files_reduce_test: list[str] = sorted(glob.glob(os.path.join(noise_path_test, 'Test_reducer_*.wav')))

    if len(list_files_normal_test) + len(list_files_abnormal_test) + len(list_files_reduce_test) != 0:
        data_mel_normal_test: list[list] = parallel_get_features_from_files(list_files_normal_test)  # 2d array
        data_mel_abnormal_test: list[list] = parallel_get_features_from_files(list_files_abnormal_test)  # 2d array
        data_mel_reduce_test: list[list] = parallel_get_features_from_files(list_files_reduce_test)  # 2d array

        print(f"[NOISE] 테스트 데이터의 수 :: {len([*data_mel_normal_test, *data_mel_abnormal_test, *data_mel_reduce_test])}")

        kph_dataset_test: list[list] = data_mel_normal_test + data_mel_abnormal_test + data_mel_reduce_test

        kph_labels_normal_test: list[int] = [0 for _ in data_mel_normal_test]
        kph_labels_abnormal_test: list[int] = [1 for _ in data_mel_abnormal_test]
        kph_labels_reduce_test: list[int] = [2 for _ in data_mel_reduce_test]
        kph_labels_test: ndarray = np.array(kph_labels_normal_test + kph_labels_abnormal_test + kph_labels_reduce_test)

        # return kph_labels_test, np.array(kph_dataset_test)
        # kph_dataset_test: ndarray = pca(kph_dataset_test,
        #                                 test_noise_before_pca_data_path,
        #                                 test_noise_after_pca_data_path,
        #                                 test_noise_pc_value_path,
        #                                 test_noise_mean_value_path,
        #                                 test_noise_pca_result_path)
        global train_mean
        global train_eigen

        t_result = []
        for file in kph_dataset_test:
            t_result.append(test_pca(file, train_mean, train_eigen))
        # pca_result = test_pca(kph_dataset_test, train_mean, train_eigen)

        t_result = np.asarray(t_result, dtype=float)

        return kph_labels_test, t_result

    return np.array([]), np.array([])


def main():
    #### Load acoustic features from wave streams

    # Training the SVM model
    train_label, train_dataset = get_train_dataset_noise()
    if len(train_dataset) != 0:
        print(f"[NOISE] Prep Data 저장 시작 -- {noise_train_prefix}/prep_data.txt")
        save_prep_data(train_dataset.tolist(), f"{noise_train_prefix}/prep_data.txt")
        print(f"[NOISE] Prep Data 저장 완료")

        train_start = time.time()
        print(f"[NOISE] 학습 시작")
        svm_model = svm_train(train_label, train_dataset, f'-c {svm_cost} -t {svm_kernel} -q')
        print(f"[NOISE] 학습 종료 - {time.time() - train_start:.3f} sec")

        # Save the trained SVM model
        svm_save_model(model_file_name=noise_model_save_path, model=svm_model)
        print(f'[NOISE] [SAVE MODEL] 모델 저장 경로 -- {noise_model_save_path}')
        # print('[NOISE][DONE] SVM model has been trained and saved')

        test_labels, test_dataset = get_test_dataset_noise()
        # save_prep_data(test_dataset.tolist(), f"{noise_train_prefix}/test_prep_data.txt")
        print(f"[NOISE] 테스트 시작")
        # Evaluation with features of the test dataset
        p_label, p_acc, p_val = svm_predict(test_labels, test_dataset, svm_model)

        # for t_label, t_data in zip(test_labels, test_dataset):
        #     test_start = time.time()
        #     svm_predict(t_label, np.expand_dims(t_data, axis=-1), svm_model)
        #     test_end = time.time()
        #     print(f"duration per file : {test_end - test_start:.3f}")

        print(f"[NOISE] 테스트 종료")
        # print(f"[NOISE] Accuracy: {p_acc}")
    else:
        print(f"[NOISE] 오류 발생 -- 학습 가능한 데이터가 없습니다.")


if __name__ == "__main__":
    main()
