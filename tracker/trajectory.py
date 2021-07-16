import numpy as np
from .object import Object

class Trajectory:
    def __init__(self,init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 tracking_features=True,
                 bb_as_features=False,
                 config = None
                 ):
        """

        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
            tracking_features: bool, if track features
            bb_as_features: bool, if treat the bb as features
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label
        self.tracking_bb_size = True
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features

        self.config = config


        self.scanning_interval = 1./self.config.LiDAR_scanning_frequency

        if self.bb_as_features:
            if self.init_features is None:
                self.init_features = init_bb
            else:
                self.init_features = np.concatenate([init_bb,init_features],0)
        self.trajectory = {}

        self.track_dim = self.compute_track_dim() # 9+4+bb_features.shape

        self.init_parameters()
        self.init_trajectory()


        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp

    def __len__(self):
        return len(self.trajectory)

    def compute_track_dim(self):
        """
        compute tracking dimension
        :return:
        """
        track_dim=9 #x,y,z,vx,vy,vz,ax,ay,az

        if self.tracking_bb_size:
            track_dim+=4 # w,h,l,yaw
        if self.tracking_features:
            track_dim+=self.init_features.shape[0] #features.shape
        return track_dim

    def init_trajectory(self):
        """
        first initialize the object state with the input boxes info,
        then initialize the trajectory with the initialized object.
        :return:
        """

        detected_state_template = np.zeros(shape=(self.track_dim-6))

        update_covariance_template = np.eye(self.track_dim)*0.01

        detected_state_template[:3] = self.init_bb[:3] #init x,y,z

        if self.tracking_bb_size:
            detected_state_template[3: 7] = self.init_bb[3:7]
            if self.tracking_features:
                detected_state_template[7: ] = self.init_features[:]
        else:
            if self.tracking_features:

                detected_state_template[3: ] = self.init_features[:]

        detected_state_template = np.mat(detected_state_template).T
        update_covariance_template = np.mat(update_covariance_template).T

        update_state_template = self.H * detected_state_template

        object = Object()

        object.updated_state = update_state_template
        object.predicted_state = update_state_template
        object.detected_state = detected_state_template
        object.updated_covariance =update_covariance_template
        object.predicted_covariance = update_covariance_template
        object.prediction_score = 1
        object.score=self.init_score
        object.features = self.init_features

        self.trajectory[self.init_timestamp] = object

    def init_parameters(self):
        """
        initialize KF tracking parameters
        :return:
        """
        self.A = np.mat(np.eye(self.track_dim))
        self.Q = np.mat(np.eye(self.track_dim))*self.config.state_func_covariance
        self.P = np.mat(np.eye(self.track_dim-6))*self.config.measure_func_covariance
        self.B = np.mat(np.zeros(shape=(self.track_dim-6,self.track_dim)))
        self.B[0:3,:] = self.A[0:3,:]
        self.B[3:,:] = self.A[9:,:]

        self.velo = np.mat(np.eye(3))*self.scanning_interval
        self.acce = np.mat(np.eye(3))*0.5*self.scanning_interval**2

        self.A[0:3,3:6] = self.velo
        self.A[3:6,6:9] = self.velo
        self.A[0:3,6:9] = self.acce

        self.H = self.B.T
        self.K = np.mat(np.zeros(shape=(self.track_dim,self.track_dim)))
        self.K[3, 0] = self.scanning_interval
        self.K[4, 1] = self.scanning_interval
        self.K[5, 2] = self.scanning_interval

    def state_prediction(self,timestamp):
        """
        predict the object state at the given timestamp
        """

        previous_timestamp = timestamp-1

        assert previous_timestamp in self.trajectory.keys()

        previous_object = self.trajectory[previous_timestamp]

        if previous_object.updated_state is not None:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
        else:
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance

        previous_prediction_score = previous_object.prediction_score

        if timestamp-1 in self.trajectory.keys():
            if self.trajectory[timestamp-1].updated_state is not None:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay*15)
            else:
                current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)
        else:
            current_prediction_score = previous_prediction_score * (1 - self.config.prediction_score_decay)


        current_predicted_state = self.A*previous_state
        current_predicted_covariance = self.A*previous_covariance*self.A.T + self.Q

        new_ob = Object()

        new_ob.predicted_state = current_predicted_state
        new_ob.predicted_covariance = current_predicted_covariance
        new_ob.prediction_score = current_prediction_score

        self.trajectory[timestamp] = new_ob
        self.consecutive_missed_num += 1

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-float(x)))

    def state_update(self,
                     bb=None,
                     features=None,
                     score=None,
                     timestamp=None,
                     ):
        """
        update the trajectory
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score:
            timestamp:
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()

        if self.bb_as_features:
            if features is None:
                features = bb
            else:
                features = np.concatenate([bb,features],0)

        detected_state_template = np.zeros(shape=(self.track_dim-6))

        detected_state_template[:3] = bb[:3] #init x,y,z

        if self.tracking_bb_size:
            detected_state_template[3: 7] = bb[3:7]
            if self.tracking_features:
                detected_state_template[7: ] = features[:]
        else:
            if self.tracking_features:
                detected_state_template[3: ] = features[:]

        detected_state_template = np.mat(detected_state_template).T

        current_ob = self.trajectory[timestamp]

        predicted_state = current_ob.predicted_state
        predicted_covariance = current_ob.predicted_covariance

        temp = self.B*predicted_covariance*self.B.T+self.P

        KF_gain = predicted_covariance*self.B.T*temp.I

        updated_state = predicted_state+KF_gain*(detected_state_template-self.B*predicted_state)
        updated_covariance = (np.mat(np.eye(self.track_dim)) - KF_gain*self.B)*predicted_covariance

        if len(self.trajectory)==2:

            updated_state = self.H*detected_state_template+\
                            self.K*(self.H*detected_state_template-self.trajectory[timestamp-1].updated_state)


        current_ob.updated_state = updated_state
        current_ob.updated_covariance = updated_covariance
        current_ob.detected_state = detected_state_template
        if self.consecutive_missed_num>1:
            current_ob.prediction_score = 1
        elif self.trajectory[timestamp - 1].updated_state is not None:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay*10*(self.sigmoid(score))
        else:
            current_ob.prediction_score = current_ob.prediction_score + self.config.prediction_score_decay*(self.sigmoid(score))
        current_ob.score = score
        current_ob.features = features

        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp

    def filtering(self,config):
        """
        filtering the trajectory in a global or near online way
        """

        wind_size = int(config.LiDAR_scanning_frequency*config.latency)

        if wind_size <0:

            detected_num = 0.00001
            score_sum = 0

            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                if ob.score is not None:
                    detected_num+=1
                    score_sum+=ob.score
                if self.first_updated_timestamp<=key<=self.last_updated_timestamp and ob.updated_state is None:
                    ob.updated_state = ob.predicted_state

            score = score_sum/detected_num
            for key in self.trajectory.keys():
                ob = self.trajectory[key]
                ob.score = score

        else:
            keys = list(self.trajectory.keys())

            for key in keys:

                min_key = int(key-wind_size)
                max_key = int(key+wind_size)
                detected_num = 0.00001
                score_sum = 0
                for key_i in range(min_key,max_key):
                    if key_i not in self.trajectory:
                        continue
                    ob = self.trajectory[key_i]
                    if ob.score is not None:
                        detected_num+=1
                        score_sum+=ob.score
                    if self.first_updated_timestamp<=key_i<=self.last_updated_timestamp and ob.updated_state is None:
                        ob.updated_state = ob.predicted_state

                score = score_sum / detected_num
                if wind_size!=0:
                    self.trajectory[key].score=score