

class Config:
    # KF parameters
    state_func_covariance = 0.01
    measure_func_covariance = 0.0001
    prediction_decay = 0.01

    LiDAR_scanning_frequency = 10

    max_prediction_num = 16
    max_prediction_num_for_new_object = 3

    # association parameters
    assign_threshold = 2.2

    # score threshold
    input_score = 0
    post_score = 1.2

config = Config()