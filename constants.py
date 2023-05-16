import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print ('Device is ', device)
# names of the sensors and number of readings for each

sensor_names = [
    "left-foot-input",
    "left-foot-output",
    "left-hip-pitch-input",
    "left-hip-roll-input",
    "left-hip-yaw-input",
    "left-knee-input",
    "left-shin-output",
    "left-tarsus-output",
    "pelvis-angular-velocity",
    "pelvis-linear-acceleration",
    "pelvis-magnetometer",
    "pelvis-orientation",
    "right-foot-input",
    "right-foot-output",
    "right-hip-pitch-input",
    "right-hip-roll-input",
    "right-hip-yaw-input",
    "right-knee-input",
    "right-shin-output",
    "right-tarsus-output",
]
# The constants are defined here
THETA_LEFT = 0.5
THETA_RIGHT = 0
MAX_STEPS = 600
OMEGA = 4.5
STEPS_IN_CYCLE = 25
a_swing = 0
b_swing = 0.5
a_stance = 0.5
b_stance = 1
FORWARD_QUARTERNIONS = np.array([1, 0, 0, 0])
KAPPA = 25
X_VEL = 2
Y_VEL = 0
Z_VEL = 0
c_swing_frc = -1
c_stance_frc = 0
c_swing_spd = 0
c_stance_spd = -1

RIGHT_FOOT = 13
LEFT_FOOT = 25
PELVIS = 1
right_foot_force_idx = 49
left_foot_force_idx = 33
# The camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,  # use the body id of Cassie
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.85)),  # adjust the height to match Cassie's height
    "elevation": -20.0,
}

actuator_ranges = {
    "left-hip-roll": [-4.5, 4.5],
    "left-hip-yaw": [-4.5, 4.5],
    "left-hip-pitch": [-12.2, 12.2],
    "left-knee": [-12.2, 12.2],
    "left-foot": [-0.9, 0.9],
    "right-hip-roll": [-4.5, 4.5],
    "right-hip-yaw": [-4.5, 4.5],
    "right-hip-pitch": [-12.2, 12.2],
    "right-knee": [-12.2, 12.2],
    "right-foot": [-0.9, 0.9],
}

exponential_bornes = {
    "q_vx": [0, X_VEL],
    "q_vy": [0, 0.5],
    "q_vz": [0, 5],
    "q_frc": [0, 1e6],
    "q_spd": [0, 2],
    "q_action": [0, 3],
    "q_orientation": [0, 1],
    "q_torque": [0, 30],
    "q_pelvis_acc": [0, 100],
}
multiplicators = {}
for key in exponential_bornes.keys():
    multiplicators[key] = -OMEGA / (
        exponential_bornes[key][1] - exponential_bornes[key][0]
    )

sensor_ranges = {
    "left-hip-roll-input": (-15, 22.5),
    "left-hip-yaw-input": (-22.5, 22.5),
    "left-hip-pitch-input": (-50, 80),
    "left-knee-input": (164, -37),
    "left-foot-input": (-140, -30),
    "left-shin-output": (-20, 20),
    "left-tarsus-output": (50, 170),
    "left-foot-output": (-140, -30),
    "right-hip-roll-input": (-22.5, 15),
    "right-hip-yaw-input": (-22.5, 22.5),
    "right-hip-pitch-input": (-50, 80),
    "right-knee-input": (-164, -37),
    "right-foot-input": (-140, -30),
    "right-shin-output": (-20, 20),
    "right-tarsus-output": (50, 170),
    "right-foot-output": (-140, -30),
    "pelvis-angular-velocity": (-34.9, 34.9),
    "pelvis-orientation": (0, 1),
}

# transform the actuator_ranges to a 2d tensor


act_ranges = torch.tensor(list(actuator_ranges.values()))

pos_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34]

vel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31]

low_obs, high_obs = torch.tensor([-3] * 23 + [-1, -1]), torch.tensor( [3] * 23 + [1, 1])
low_obs = low_obs
high_obs = high_obs
low_action, high_action = [], []
for key in actuator_ranges.keys():
    low_action.append(actuator_ranges[key][0])
    high_action.append(actuator_ranges[key][1])

low_action = torch.tensor(low_action)
high_action = torch.tensor(high_action)


obs_ranges = np.array([[ -11.,  -20.,  -19.,  -50., -136.,   -1.,    0.,   -3.,  -16.,
         -18.,  -19.,  -51., -133.,   -1.,    0.,   -3.,  -15.,  -26.,
         -16.,   -1.,   -1.,   -1.,   -1.,   -1.,   -1.],
       [  15.,   16.,   28.,   -4.,   -2.,    1.,    4.,   -0.,   12.,
          18.,   28.,   -4.,   -2.,    1.,    4.,   -0.,   15.,   27.,
          16.,    1.,    1.,    1.,    1.,    1.,    1.]])


sensor_obs_ranges= np.array([[ -12.,  -17.,  -19.,  -50., -131.,   -1.,    0.,   -3.,  -15.,
         -17.,  -20.,  -50., -133.,   -1.,    0.,   -3.,   -1.,   -1.,
          -1.,   -1.,  -15.,  -30.,  -15., -157., -157., -157.,   -1.,
          -1.,   -1.],
       [  16.,   18.,   27.,   -4.,   -1.,    1.,    4.,   -0.,   12.,
          18.,   28.,   -4.,   -2.,    1.,    4.,   -0.,    1.,    1.,
           1.,    1.,   14.,   29.,   13.,  157.,  157.,  157.,    1.,
           1.,    1.]])

