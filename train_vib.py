# Data loading
import glob
import os

# Python libraries
import numpy as np
# Training
from libsvm.svmutil import svm_train, svm_save_model, svm_predict
from numpy import ndarray

import Constants
from Constants import path_train
from PCA_Loader import pca
from VibUtil import parallel_get_features_from_files_vib


#### Import libraries
def get_test_dataset_vibe() -> tuple[ndarray, list[list]]:
    print('[VIBE][LOAD] Extracting features from test dataset: {}'.format(Constants.path_test))
    list_files_normal_test = glob.glob(os.path.join(Constants.path_test, 'normal_*.asc'))
    list_files_anomal_test = glob.glob(os.path.join(Constants.path_test, 'anomal_*.asc'))
    list_files_reduce_test = glob.glob(os.path.join(Constants.path_test, 'reducer_*.asc'))
    list_files_normal_test.sort()
    list_files_anomal_test.sort()
    list_files_reduce_test.sort()
    if len(list_files_normal) + len(list_files_anomal) + len(list_files_reduce) != 0:
        data_mel_normal_test = parallel_get_features_from_files_vib(list_files_normal_test)
        data_mel_anomal_test = parallel_get_features_from_files_vib(list_files_anomal_test)
        data_mel_reduce_test = parallel_get_features_from_files_vib(list_files_reduce_test)

        kph_dataset_test = [*data_mel_normal_test, *data_mel_anomal_test, *data_mel_reduce_test]

        kph_labels_normal_test = np.zeros(len(data_mel_normal_test), dtype=int)
        kph_labels_anomal_test = np.ones(len(data_mel_anomal_test), dtype=int)
        kph_labels_reduce_test = np.ones(len(data_mel_reduce_test), dtype=int) * 2
        kph_labels_test = np.array(
            list(kph_labels_normal_test) + list(kph_labels_anomal_test) + list(kph_labels_reduce_test))

        return (kph_labels_test, kph_dataset_test)

    return ([], [])


def get_train_dataset_vibe() -> tuple:
    # Train dataset
    print('[LOAD] Extracting features from train dataset: {}'.format(path_train))

    list_files_normal = glob.glob(os.path.join(path_train, "normal_*.asc"))
    list_files_anomal = glob.glob(os.path.join(path_train, "anomal_*.asc"))
    list_files_reduce = glob.glob(os.path.join(path_train, "reducer_*.asc"))
    list_files_normal.sort()
    list_files_anomal.sort()
    list_files_reduce.sort()

    if len(list_files_normal) + len(list_files_anomal) + len(list_files_reduce) != 0:
        data_mel_normal = parallel_get_features_from_files_vib(list_files_normal)
        data_mel_anomal = parallel_get_features_from_files_vib(list_files_anomal)
        data_mel_reduce = parallel_get_features_from_files_vib(list_files_reduce)

        # Merge normal and abnormal data
        kph_dataset = data_mel_normal + data_mel_anomal + data_mel_reduce
        # Make labels
        kph_labels_normal = np.zeros(len(data_mel_normal), dtype=int)
        kph_labels_anomal = np.ones(len(data_mel_anomal), dtype=int)
        kph_labels_reduce = np.ones(len(data_mel_reduce), dtype=int) * 2
        kph_labels = np.array(list(kph_labels_normal) + list(kph_labels_anomal) + list(kph_labels_reduce))

        kph_dataset = pca(kph_dataset, Constants.vibe_before_pca_data_path, Constants.vibe_after_pca_data_path)
        print(f'{len(kph_dataset)} files are loaded from the train dataset')
        return (kph_labels, kph_dataset)

    return ([], [])


def main():
    # Training the SVM model
    train_label, train_dataset = get_train_dataset_vibe()

    if len(train_label) + len(train_dataset) != 0:
        print(f"[VIBE] 학습 시작")
        svm_model = svm_train(train_label, train_dataset, f'-c {Constants.svm_cost} -t {Constants.svm_kernel} -q')
        print(f"[VIBE] 학습 종료")
        # Save the trained SVM model
        svm_save_model(model_file_name=Constants.vib_model_save_path, model=svm_model)

        print('[VIBE][DONE] SVM model has been trained and saved')

        # test_label, test_dataset = get_test_dataset_vibe()
        #
        # # Evaluation with features of the test dataset
        # p_label, p_acc, p_val = svm_predict(test_label, test_dataset, svm_model)
        # print('Vibe Accuracy: {}'.format(p_val))
    else:
        print(f"[VIBE] 오류 발생")


if __name__ == "__main__":
    main()
