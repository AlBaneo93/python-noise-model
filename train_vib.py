# Data loading
import glob
import os

# Python libraries
import numpy as np
# Training
from libsvm.svmutil import svm_train, svm_save_model, svm_predict
from numpy import ndarray

from Constants import path_train, vibe_before_pca_data_path, vibe_after_pca_data_path, vibe_pc_value_path, \
    vibe_mean_value_path, vibe_pca_result_path, test_vibe_before_pca_data_path, test_vibe_after_pca_data_path, \
    test_vibe_pc_value_path, test_vibe_mean_value_path, test_vibe_pca_result_path, svm_cost, svm_kernel, \
    vibe_model_save_path, path_test
from PCA_Loader import pca
from VibUtil import parallel_get_features_from_files_vib


def get_train_dataset_vibe() -> tuple[ndarray, ndarray]:
    # Train dataset
    print('[LOAD] Extracting features from train dataset: {}'.format(path_train))

    list_files_normal = sorted(glob.glob(os.path.join(path_train, "normal_*.asc")))
    list_files_abnormal = sorted(glob.glob(os.path.join(path_train, "anomal_*.asc")))
    list_files_reduce = sorted(glob.glob(os.path.join(path_train, "reducer_*.asc")))

    if len(list_files_normal) + len(list_files_abnormal) + len(list_files_reduce) != 0:
        data_mel_normal = parallel_get_features_from_files_vib(list_files_normal)
        data_mel_abnormal = parallel_get_features_from_files_vib(list_files_abnormal)
        data_mel_reduce = parallel_get_features_from_files_vib(list_files_reduce)

        # Merge normal and abnormal data
        kph_dataset = data_mel_normal + data_mel_abnormal + data_mel_reduce
        # Make labels
        kph_labels_normal = np.zeros(len(data_mel_normal), dtype=int)
        kph_labels_abnormal = np.ones(len(data_mel_abnormal), dtype=int)
        kph_labels_reduce = np.ones(len(data_mel_reduce), dtype=int) * 2
        kph_labels = np.array(list(kph_labels_normal) + list(kph_labels_abnormal) + list(kph_labels_reduce))

        kph_dataset_reshaped: ndarray = pca(kph_dataset,
                                            vibe_before_pca_data_path,
                                            vibe_after_pca_data_path,
                                            vibe_pc_value_path,
                                            vibe_mean_value_path,
                                            vibe_pca_result_path
                                            )
        print(f'{len(kph_dataset_reshaped)} files are loaded from the train dataset')
        return kph_labels, kph_dataset_reshaped

    return np.array([]), np.array([])


def get_test_dataset_vibe() -> tuple[ndarray, ndarray]:
    print('[VIBE][LOAD] Extracting features from test dataset: {}'.format(path_test))
    list_files_normal_test = sorted(glob.glob(os.path.join(path_test, 'Test_normal_*.asc')))
    list_files_abnormal_test = sorted(glob.glob(os.path.join(path_test, 'Test_anomal_*.asc')))
    list_files_reduce_test = sorted(glob.glob(os.path.join(path_test, 'Test_reducer_*.asc')))

    if len(list_files_normal_test) + len(list_files_abnormal_test) + len(list_files_reduce_test) != 0:
        data_mel_normal_test = parallel_get_features_from_files_vib(list_files_normal_test)
        data_mel_abnormal_test = parallel_get_features_from_files_vib(list_files_abnormal_test)
        data_mel_reduce_test = parallel_get_features_from_files_vib(list_files_reduce_test)

        kph_dataset_test = data_mel_normal_test + data_mel_abnormal_test + data_mel_reduce_test

        kph_labels_normal_test = [0 for _ in data_mel_normal_test]
        kph_labels_abnormal_test = [1 for _ in data_mel_abnormal_test]
        kph_labels_reduce_test = [2 for _ in data_mel_reduce_test]

        kph_labels_test: ndarray = np.array(kph_labels_normal_test + kph_labels_abnormal_test + kph_labels_reduce_test)

        # TODO : 테스트 데이터에 대한 경로 지정하기

        kph_dataset_test_reshaped: ndarray = pca(kph_dataset_test,
                                                 test_vibe_before_pca_data_path,
                                                 test_vibe_after_pca_data_path,
                                                 test_vibe_pc_value_path,
                                                 test_vibe_mean_value_path,
                                                 test_vibe_pca_result_path)

        return kph_labels_test, kph_dataset_test_reshaped

    return np.array([]), np.array([])


def main():
    # Training the SVM model
    train_label, train_dataset = get_train_dataset_vibe()

    if len(train_dataset) != 0:
        print(f"[VIBE] 학습 시작")
        svm_model = svm_train(train_label, train_dataset, f'-c {svm_cost} -t {svm_kernel} -q')
        print(f"[VIBE] 학습 종료")
        # Save the trained SVM model
        svm_save_model(model_file_name=vibe_model_save_path, model=svm_model)

        print('[VIBE][DONE] SVM model has been trained and saved')

        test_label, test_dataset = get_test_dataset_vibe()

        # Evaluation with features of the test dataset
        p_label, p_acc, p_val = svm_predict(test_label, test_dataset, svm_model)
        print('Vibe Accuracy: {}'.format(p_acc))
    else:
        print(f"[VIBE] 오류 발생")


if __name__ == "__main__":
    main()
