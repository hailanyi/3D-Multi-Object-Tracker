

class Config:
    # KF parameters
    state_func_covariance = 0.01
    measure_func_covariance = 0.0001
    prediction_score_decay = 0.03

    LiDAR_scanning_frequency = 10

    # max prediction number of state function
    max_prediction_num = 16
    max_prediction_num_for_new_object = 3

    # detection score threshold
    input_score = 0
    post_score = 3

    # globally filtering
    globally = True


config = Config()