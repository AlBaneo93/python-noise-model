# For Noise
n_ffts = 2048
n_step = 1024
n_wins = 1024
n_mels = 16  # 64
n_mfcc = 20
noise_sr = 22050
noise_pca_data_path = "/data/noise_before_pca.txt"
noise_after_pca_data_path = "/data/noise_after_pca.txt"
audio_loader = ["java", "-jar", "/data/AudioLoader_New.jar"]
noise_model_save_path = "/data/noise_svm_trained.model"
resample_result_path = "/data/resample_result"
noise_offset = 0
noise_cut_time = 8

# For VIBE
vib_model_save_path = "/data/vib_svm_trained.model"
vibe_sr = 2048
vibe_pca_data_path = "/data/vibe_before_pca.txt"
vibe_after_pca_data_path = "/data/vibe_after_pca.txt"
vibe_offset = 0
vibe_cut_time = 8

# Common
path_train = "/data/CS_MOTORNOISE_DATA"
path_test = "/data/CS_MORTORNOISE_TEST_DATA"

svm_cost = 1
svm_type = 0
svm_kernel = 0

pca_loader = ["java", "-jar", "/data/PCA.jar"]
num_components = 50
