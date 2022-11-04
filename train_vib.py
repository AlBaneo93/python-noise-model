from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#### Import libraries

# Python libraries
import argparse
import numpy as np
import os
import sys

# Data loading
import csv
import glob
import pickle
import yaml

# Training
# from sklearn import svm
from libsvm.svmutil import svm_train, svm_predict, svm_save_model

import Constants
from Constants import vib_svm_model, path_train, path_test
from VibUtil import get_features_from_files_vib
from util import reshape_features


def main():
    print("It is Sunday...")

    #### Load acoustic features from asce streams

    # Train dataset
    print('[LOAD] Extracting features from train dataset: {}'.format(path_train))
    train_dir_data = path_train  # "./dataset/java/CS_MOTORNOISE_DATA/train"
    list_files_normal = glob.glob(os.path.join(train_dir_data, '*.normal_*.asc'))
    list_files_anomal = glob.glob(os.path.join(train_dir_data, '*.anomal_*.asc'))
    list_files_reduce = glob.glob(os.path.join(train_dir_data, '*.reducer_*.asc'))
    list_files_normal.sort()
    list_files_anomal.sort()
    list_files_reduce.sort()

    data_mel_normal = get_features_from_files_vib(list_files_normal)
    data_mel_anomal = get_features_from_files_vib(list_files_anomal)
    data_mel_reduce = get_features_from_files_vib(list_files_reduce)

    # Merge normal and abnormal data
    kph_dataset = data_mel_normal + data_mel_anomal + data_mel_reduce

    # Make labels
    kph_labels_normal = np.zeros(len(data_mel_normal), dtype=int)
    kph_labels_anomal = np.ones(len(data_mel_anomal), dtype=int)
    kph_labels_reduce = np.ones(len(data_mel_reduce), dtype=int) * 2
    kph_labels = np.array(list(kph_labels_normal) + list(kph_labels_anomal) + list(kph_labels_reduce))

    print(f'{len(kph_dataset)} files are loaded from the train dataset')

    # Test dataset

    print('[LOAD] Extracting features from test dataset: {}'.format(path_test))
    test_dir_data = path_test  # "./dataset/java/CS_MOTORNOISE_DATA/test"
    list_files_normal_test = glob.glob(os.path.join(test_dir_data, '*.normal_*.asc'))
    list_files_anomal_test = glob.glob(os.path.join(test_dir_data, '*.anomal_*.asc'))
    list_files_reduce_test = glob.glob(os.path.join(test_dir_data, '*.reducer_*.asc'))
    list_files_normal_test.sort()
    list_files_anomal_test.sort()
    list_files_reduce_test.sort()

    data_mel_normal_test = get_features_from_files_vib(list_files_normal_test)
    data_mel_anomal_test = get_features_from_files_vib(list_files_anomal_test)
    data_mel_reduce_test = get_features_from_files_vib(list_files_reduce_test)

    kph_dataset_test = [*data_mel_normal_test, *data_mel_anomal_test, *data_mel_reduce_test]

    kph_labels_normal_test = np.zeros(len(data_mel_normal_test), dtype=int)
    kph_labels_anomal_test = np.ones(len(data_mel_anomal_test), dtype=int)
    kph_labels_reduce_test = np.ones(len(data_mel_reduce_test), dtype=int) * 2
    kph_labels_test = np.array(
        list(kph_labels_normal_test) + list(kph_labels_anomal_test) + list(kph_labels_reduce_test))

    #### Reshape and Scaling

    kph_dataset_reshaped = kph_dataset
    kph_dataset_test_reshaped = kph_dataset_test

    #### Training the features using SVM

    # Training the SVM model
    svm_cost = int(Constants.svm_cost)
    svm_kernel = int(Constants.svm_kernel)

    svm_model = svm_train(kph_labels, kph_dataset_reshaped, f'-c {svm_cost} -t {svm_kernel} -q')

    # Save the trained SVM model
    svm_save_model(vib_svm_model, svm_model)

    print('[DONE] SVM model has been trained and saved')

    # Evaluation with features of the test dataset
    p_label, p_acc, p_val = svm_predict(kph_labels_test, kph_dataset_test_reshaped, svm_model)
    print('Accuracy: {}'.format(p_val))


if __name__ == "__main__":
    main()
