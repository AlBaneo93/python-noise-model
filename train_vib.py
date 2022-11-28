# Data loading
import datetime
import glob
import os
import time

# Python libraries
import numpy as np
# Training
from libsvm.svmutil import svm_train, svm_save_model, svm_predict
from numpy import ndarray

from CommonUtils import save_prep_data, apply_pca, test_pca
from Constants import svm_cost, svm_kernel, \
    vibe_model_save_path, vibe_train_prefix, vibe_path_test, vibe_path_train, vibe_mean_value_path, vibe_pc_value_path, \
    test_vibe_mean_value_path, test_vibe_pc_value_path
from VibUtil import parallel_get_features_from_files_vib

train_mean = np.array([])
train_eigen = np.array([])


def get_train_dataset_vibe() -> tuple[ndarray, ndarray]:
    # Train dataset
    print('[VIBE] [LOAD] Extracting features from train dataset: {}'.format(vibe_path_train))

    list_files_normal = sorted(glob.glob(os.path.join(vibe_path_train, "normal_*.asc")))
    list_files_abnormal = sorted(glob.glob(os.path.join(vibe_path_train, "anomal_*.asc")))
    # list_files_reduce = sorted(glob.glob(os.path.join(vibe_path_train, "reducer_*.asc")))

    if len([*list_files_normal, *list_files_abnormal]) != 0:
        data_mel_normal = parallel_get_features_from_files_vib(list_files_normal)
        data_mel_abnormal = parallel_get_features_from_files_vib(list_files_abnormal)
        # data_mel_reduce = parallel_get_features_from_files_vib(list_files_reduce)

        print(f"[VIBE] 학습 데이터의 수 :: {len([*data_mel_normal, *data_mel_abnormal])}")
        # Merge normal and abnormal data
        kph_dataset = data_mel_normal + data_mel_abnormal

        # Make labels
        kph_labels_normal = np.zeros(len(data_mel_normal), dtype=int)
        kph_labels_abnormal = np.ones(len(data_mel_abnormal), dtype=int)
        # kph_labels_reduce = np.ones(len(data_mel_reduce), dtype=int) * 2
        kph_labels = np.asarray(list(kph_labels_normal) + list(kph_labels_abnormal))

        # return kph_labels, np.array(kph_dataset)
        # kph_dataset_reshaped: ndarray = pca(kph_dataset,
        #                                     vibe_before_pca_data_path,
        #                                     vibe_after_pca_data_path,
        #                                     vibe_pc_value_path,
        #                                     vibe_mean_value_path,
        #                                     vibe_pca_result_path
        #                                     )
        # print(f'{len(kph_dataset_reshaped)} files are loaded from the train dataset')
        pca_result, mean, eigen = apply_pca(kph_dataset, vibe_mean_value_path, vibe_pc_value_path)

        global train_mean
        global train_eigen

        train_mean = np.asarray(mean)
        train_eigen = np.asarray(eigen)
        return kph_labels, pca_result

    return np.array([]), np.array([])


def get_test_dataset_vibe() -> tuple[ndarray, ndarray]:
    print('[VIBE] [LOAD] Extracting features from test dataset: {}'.format(vibe_path_test))
    list_files_normal_test = sorted(glob.glob(os.path.join(vibe_path_test, 'Test_normal_*.asc')))
    list_files_abnormal_test = sorted(glob.glob(os.path.join(vibe_path_test, 'Test_anomal_*.asc')))
    # list_files_reduce_test = sorted(glob.glob(os.path.join(vibe_path_test, 'Test_reducer_*.asc')))

    if len(list_files_normal_test) + len(list_files_abnormal_test) != 0:
        data_mel_normal_test = parallel_get_features_from_files_vib(list_files_normal_test)
        data_mel_abnormal_test = parallel_get_features_from_files_vib(list_files_abnormal_test)
        # data_mel_reduce_test = parallel_get_features_from_files_vib(list_files_reduce_test)

        print(f"[VIBE] 테스트 데이터의 수 :: {len([*data_mel_normal_test, *data_mel_abnormal_test])}")

        kph_dataset_test = data_mel_normal_test + data_mel_abnormal_test

        kph_labels_normal_test = [0 for _ in data_mel_normal_test]
        kph_labels_abnormal_test = [1 for _ in data_mel_abnormal_test]
        # kph_labels_reduce_test = [2 for _ in data_mel_reduce_test]

        kph_labels_test: ndarray = np.asarray(kph_labels_normal_test + kph_labels_abnormal_test)

        # return kph_labels_test, np.array(kph_dataset_test)

        # kph_dataset_test_reshaped: ndarray = pca(kph_dataset_test,
        #                                          test_vibe_before_pca_data_path,
        #                                          test_vibe_after_pca_data_path,
        #                                          test_vibe_pc_value_path,
        #                                          test_vibe_mean_value_path,
        #                                          test_vibe_pca_result_path)
        #
        global train_mean
        global train_eigen
        # print(f"data shape : {np.asarray(kph_dataset_test[0]).shape}")
        # print(f"mean shape : {train_mean.shape} / eigen shape : {train_eigen.shape}")

        pca_result = test_pca(kph_dataset_test, train_mean, train_eigen)
        return kph_labels_test, pca_result

    return np.array([]), np.array([])


def main():
    # Training the SVM model
    train_label, train_dataset = get_train_dataset_vibe()

    if len(train_dataset) != 0:

        print(f"[VIBE] Prep Data 저장 시작 -- {vibe_train_prefix}/prep_data.txt")
        save_prep_data(train_dataset.tolist(), f"{vibe_train_prefix}/prep_data.txt")
        print(f"[VIBE] Prep Data 저장 완료")

        train_start = time.time()
        print(f"[VIBE] 학습 시작")
        svm_model = svm_train(train_label, train_dataset, f'-c {svm_cost} -t {svm_kernel} -q')
        print(f"[VIBE] 학습 종료 - {time.time() - train_start:.3f} sec")

        # Save the trained SVM model
        svm_save_model(model_file_name=vibe_model_save_path, model=svm_model)
        print(f'[VIBE] [SAVE MODEL] 모델 저장 경로 -- {vibe_model_save_path}')
        # print('[VIBE][DONE] SVM model has been trained and saved')

        test_label, test_dataset = get_test_dataset_vibe()
        save_prep_data(test_dataset.tolist(), f"{vibe_train_prefix}/test_prep_data.txt")
        # Evaluation with features of the test dataset


        a = 1
        print(f"[VIBE] 테스트 시작")
        p_label, p_acc, p_val = svm_predict(test_label, test_dataset, svm_model)

        # for t_label, t_data in zip(test_label, test_dataset):
        #     test_start = time.time()
        #     svm_predict(t_label, t_data, svm_model)
        #     test_end = time.time()
        #     print(f"duration per file : {test_end - test_start:.3f}")

        print(f"[VIBE] 테스트 종료")
        # print(f"[VIBE] Accuracy: {p_acc}")
    else:
        print(f"[VIBE] 오류 발생 -- 학습 가능한 데이터가 없습니다. {len(train_dataset)}")


if __name__ == "__main__":
    main()
