# Data loading
import glob
import os

# Python libraries
import numpy as np
from libsvm.svmutil import svm_train, svm_save_model, svm_predict

import Constants
from PCA_Loader import pca
# Sound processing
from util import parallel_get_features_from_files


#### Import libraries
def get_train_dataset_noise() -> tuple:
    print('[NOISE][LOAD] Extracting features from train dataset: {}'.format(Constants.path_train))
    list_files_normal = glob.glob(os.path.join(Constants.path_train, 'normal_*.wav'))
    list_files_anomal = glob.glob(os.path.join(Constants.path_train, 'anomal_*.wav'))
    list_files_reduce = glob.glob(os.path.join(Constants.path_train, 'reducer_*.wav'))
    list_files_normal.sort()
    list_files_anomal.sort()
    list_files_reduce.sort()

    if len(list_files_normal) + len(list_files_anomal) + len(list_files_reduce) != 0:
        print(f"[NOISE] 데이터 추출 시작")
        # NOTE : 2D Array이어야 함
        data_mel_normal = parallel_get_features_from_files(list_files_normal)
        # data_mel_normal = get_features_from_files(list_files_normal)
        # NOTE : 2D Array이어야 함
        data_mel_anomal = parallel_get_features_from_files(list_files_anomal)
        # data_mel_anomal = get_features_from_files(list_files_anomal)
        # NOTE : 2D Array이어야 함
        data_mel_reduce = parallel_get_features_from_files(list_files_reduce)
        # data_mel_reduce = get_features_from_files(list_files_reduce)
        print(f"[NOISE] 데이터 추출 완료")

        # Merge normal and abnormal data
        # kph_dataset = data_mel_normal + data_mel_anomal + data_mel_reduce
        kph_dataset = data_mel_normal + data_mel_anomal + data_mel_reduce  # 2d array
        # print(f"전체 데이터셋 크기 : {len(kph_dataset)}")
        # print(f"Apply PCA")
        # kph_dataset = pca(kph_dataset)  # 전체 데이터에 PCA 적용
        # print(f"Train dataset size : {len(kph_dataset)}")

        # Make labels
        kph_labels_normal = np.zeros(len(data_mel_normal), dtype=int)
        kph_labels_anomal = np.ones(len(data_mel_anomal), dtype=int)
        kph_labels_reduce = np.ones(len(data_mel_reduce), dtype=int) * 2
        kph_labels = np.array(list(kph_labels_normal) + list(kph_labels_anomal) + list(kph_labels_reduce))

        # reshape the train dataset
        # kph_dataset_reshaped = reshape_features(kph_dataset)
        # NOTE : 2D array여야 함
        kph_dataset_reshaped = pca(kph_dataset, Constants.noise_before_pca_data_path,
                                   Constants.noise_after_pca_data_path)

        return (kph_labels, kph_dataset_reshaped)

    return ([], [])


def get_test_dataset_noise() -> tuple:
    # Test dataset
    print('[NOISE][LOAD] Extracting features from test dataset: {}'.format(Constants.path_test))

    list_files_normal_test = glob.glob(os.path.join(Constants.path_test, 'normal_*.wav'))
    list_files_anomal_test = glob.glob(os.path.join(Constants.path_test, 'anomal_*.wav'))
    list_files_reduce_test = glob.glob(os.path.join(Constants.path_test, 'reducer_*.wav'))
    list_files_normal_test.sort()
    list_files_anomal_test.sort()
    list_files_reduce_test.sort()

    if len(list_files_normal) + len(list_files_anomal) + len(list_files_reduce) != 0:
        data_mel_normal_test = parallel_get_features_from_files(list_files_normal_test)  # 2d array
        data_mel_anomal_test = parallel_get_features_from_files(list_files_anomal_test)  # 2d array
        data_mel_reduce_test = parallel_get_features_from_files(list_files_reduce_test)  # 2d array

        kph_dataset_test = data_mel_normal_test + data_mel_anomal_test + data_mel_reduce_test
        kph_labels_normal_test = np.zeros(len(data_mel_normal_test), dtype=int)
        kph_labels_anomal_test = np.ones(len(data_mel_anomal_test), dtype=int)
        kph_labels_reduce_test = np.ones(len(data_mel_reduce_test), dtype=int) * 2
        kph_labels_test = np.array(
            list(kph_labels_normal_test) + list(kph_labels_anomal_test) + list(kph_labels_reduce_test))
        return (kph_labels_test, kph_dataset_test)

    return ([], [])


def main():
    #### Load acoustic features from wave streams

    # Training the SVM model
    train_label, train_dataset = get_train_dataset_noise()
    if len(train_label) + len(train_dataset) != 0:
        print(f"[NOISE] 학습 시작")
        svm_model = svm_train(train_label, train_dataset, f'-c {Constants.svm_cost} -t {Constants.svm_kernel} -q')
        print(f"[NOISE] 학습 종료")

        # Save the trained SVM model
        svm_save_model(model_file_name=Constants.noise_model_save_path, model=svm_model)

        print('[NOISE][DONE] SVM model has been trained and saved')

        # test_labels, test_dataset = get_test_dataset_noise()
        #
        # # Evaluation with features of the test dataset
        # p_label, p_acc, p_val = svm_predict(test_labels, test_dataset, svm_model)
        # print('[NOISE] Accuracy: {}'.format(p_val))
    else:
        print(f"[NOISE] 오류 발생")


if __name__ == "__main__":
    main()
