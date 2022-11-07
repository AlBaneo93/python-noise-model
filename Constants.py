n_ffts = 2048
n_step = 1024
n_wins = 1024
n_mels = 16  # 64
n_mfcc = 20
# y_scale = True
# y_mono = True
sr = 22050
win_size = 4
step_size = 2
svm_cost = 1
svm_type = 0
svm_kernel = 0

# path_train = "/data/CS_MOTORNOISE_DATA"
path_train = "/data/test"
path_test = "/data/CS_MORTORNOISE_TEST_DATA"
# svm_model = "/data/total_file_pca_svm_trained.model"
noise_model_save_path = "/data/total_file_pca_svm_trained.model"
vib_model_save_path = "/data/vib_svm_trained.model"
vibe_sr = 2048
audio_loader = ["java", "-jar", "/data/AudioLoader_New.jar"]
pca_loader = ["java", "-jar", "/data/PCA.jar"]
resample_result_path = "/data/resample_result"
num_components = 2
pca_data_path = "/data/before_pca.txt"

noise_offset = 0
noise_cut_time = 1
vibe_cut_time = 1
vibe_offset = 0
