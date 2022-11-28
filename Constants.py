# Common
noise_path_train = "/data/CS_MOTORNOISE_DATA"
noise_path_test = "/data/TEST_CS_MOTORNOISE_DATA"

svm_cost = 1
svm_type = 0
svm_kernel = 0
num_components = 50

pca_loader = ["java", "-jar", "/data/PCA.jar"]


# For Noise Common
n_ffts = 2048
n_step = 1024
n_wins = 1024
n_mels = 64  # 64
n_mfcc = 20
noise_sr = 22050
noise_offset = 0.25
noise_cut_time = 8

audio_loader = ["java", "-jar", "/data/AudioLoader_New.jar"]

resample_result_path = "/data/resample_result"

# For Noise Train
noise_train_prefix = "/data/noise/train"

noise_before_pca_data_path = f"{noise_train_prefix}/before_pca_{num_components}.txt"
noise_after_pca_data_path = f"{noise_train_prefix}/after_pca_{num_components}.txt"
noise_pc_value_path = f"{noise_train_prefix}/pc_value_{num_components}.txt"
noise_mean_value_path = f"{noise_train_prefix}/mean_value_{num_components}.txt"
noise_pca_result_path = f"{noise_train_prefix}/pca_result_{num_components}.txt"

noise_model_save_path = f"/data/results/noise_svm_trained_{num_components}.model"

# For Noise Test
noise_test_prefix = "/data/noise/test"

test_noise_before_pca_data_path = f"{noise_test_prefix}/before_pca_{num_components}.txt"
test_noise_after_pca_data_path = f"{noise_test_prefix}/after_pca_{num_components}.txt"
test_noise_pc_value_path = f"{noise_test_prefix}/pc_value_{num_components}.txt"
test_noise_mean_value_path = f"{noise_test_prefix}/mean_value_{num_components}.txt"
test_noise_pca_result_path = f"{noise_test_prefix}/pca_result_{num_components}.txt"

# ---------------------------------------------------------------------------------------------------

# For Vibe Common
vibe_path_train="/data/QZ_FCEV_IDLE_DATA"
vibe_path_test ="/data/TEST_QZ_FCEV_IDLE_DATA"

vibe_sr = 2048
vibe_offset = 0.25
vibe_cut_time = 8

# For VIBE Train

vibe_train_prefix = "/data/vibe/train"

vibe_before_pca_data_path = f"{vibe_train_prefix}/before_pca_{num_components}.txt"
vibe_after_pca_data_path = f"{vibe_train_prefix}/after_pca_{num_components}.txt"
vibe_pc_value_path = f"{vibe_train_prefix}/pc_value_{num_components}.txt"
vibe_mean_value_path = f"{vibe_train_prefix}/mean_value_{num_components}.txt"
vibe_pca_result_path = f"{vibe_train_prefix}/pca_result_{num_components}.txt"

vibe_model_save_path = f"/data/results/vibe_svm_trained_{num_components}.model"

# For VIBE Test
vibe_test_prefix = "/data/vibe/test"

test_vibe_before_pca_data_path = f"{vibe_test_prefix}/before_pca_{num_components}.txt"
test_vibe_after_pca_data_path = f"{vibe_test_prefix}/after_pca_{num_components}.txt"
test_vibe_pc_value_path = f"{vibe_test_prefix}/pc_value_{num_components}.txt"
test_vibe_mean_value_path = f"{vibe_test_prefix}/mean_value_{num_components}.txt"
test_vibe_pca_result_path = f"{vibe_test_prefix}/pca_result_{num_components}.txt"
