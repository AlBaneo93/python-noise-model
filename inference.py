# #### Import libraries
#
# # Python libraries
# import argparse
# import numpy as np
# import os
# import sys
#
# # Data loading
# import glob
# import pickle
# import yaml
#
# # Preprocessing
# from sklearn.decomposition import PCA
#
# import Constants
# from util import reshape_features, get_features_from_files
#
# # Training
# from libsvm.svmutil import svm_load_model, svm_predict
#
# # # Evaluation
# # from sklearn.metrics import confusion_matrix
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
#
# from Constants import *
#
# def main():
#     print ("It is Sunday afternoon...")
#
#     # Load the test dataset
#     # if os.path.exists(args.feat_test):
#     #     try:
#     #         dataset = np.load(args.feat_test)
#     #         data = dataset['data']
#     #         label = dataset['label']
#     #     except:
#     #         sys.exit("[ERR] The loaded feature data is not valid")
#     # else:
#     #     print('[LOAD] Extracting features from test dataset: {}'.format(path_test))
#
#     if os.path.exists(path_test):
#         try:
#             dir_data = path_test
#             list_files_normal_test = glob.glob(os.path.join(dir_data, 'normal_*.wav'))
#             list_files_abnormal_test = glob.glob(os.path.join(dir_data, 'abnormal_*.wav'))
#             list_files_reduce_test = glob.glob(os.path.join(dir_data, 'reducer_*.wav'))
#             list_files_normal_test.sort()
#             list_files_abnormal_test.sort()
#             list_files_reduce_test.sort()
#
#             data_mel_normal_test = get_features_from_files(list_files_normal_test)
#             data_mel_abnormal_test = get_features_from_files(list_files_abnormal_test)
#             data_mel_reduce_test = get_features_from_files(list_files_reduce_test)
#             data = data_mel_normal_test + data_mel_abnormal_test + data_mel_reduce_test
#
#             kph_labels_normal_test = np.zeros(len(data_mel_normal_test), dtype=int)
#             kph_labels_abnormal_test = np.ones(len(data_mel_abnormal_test), dtype=int)
#             kph_labels_reduce_test = np.ones(len(data_mel_reduce_test), dtype=int) * 2
#             label = np.array(list(kph_labels_normal_test) + list(kph_labels_abnormal_test) + list(kph_labels_reduce_test))
#
#             # Save the extracted features
#             np.savez('feature_mel_test.npz', data=data, label=label)
#         except:
#             sys.exit("[ERR] The directory path is not valid")
#     else:
#         sys.exit("[ERR] Test dataset is not given")
#
#     print(f'{len(data)} files are loaded from the test dataset')
#
#
#     #### Reshape and Scaling
#     if os.path.exists(scaler_model):
#         try:
#             with open(scaler_model, 'rb') as pickle_file:
#                 scaler = pickle.load(pickle_file)
#                 data_scaled = scaler.transform(reshape_features(data))
#             print('[DONE] Reshape and scale')
#         except:
#             sys.exit("[ERR] The loaded scaler is not valid")
#     else:
#         sys.exit("[ERR] Scaler path is not given")
#
#
#     #### Dimensionality reduction using PCA
#     # if os.path.exists(pca_model):
#     #     try:
#     #         with open(pca_model, 'rb') as pickle_file:
#     #             pca_model = pickle.load(pickle_file)
#     #             data_pcs = pca_model.transform(data_scaled)
#     #         print('[DONE] Dimensionality reduction')
#     #     except:
#     #         sys.exit("[ERR] The loaded PCA model is not valid")
#     # else:
#     #     sys.exit("[ERR] No fitted PCA model is given")
#
#
#     #### Evaluation with features of the test dataset
#     if os.path.exists(Constants.svm_model):
#         try:
#             svm_model = svm_load_model(Constants.svm_model)
#             p_label, p_acc, p_val = svm_predict(label, reshape_features(data), svm_model)
#         except:
#             sys.exit("[ERR] The loaded SVM model is not valid")
#     else:
#         sys.exit("[ERR] No trained SVM model is given")
#
#
# if __name__ == "__main__":
#     main()
