from collections import OrderedDict
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
mass = 33.8502
gravity = 9.81

# names of the sensors and number of readings for each
caps_lamda_s = 1000
caps_lamda_t = 1
caps_sigma = 0.01
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

FORWARD_QUARTERNIONS = np.array([1, 0, 0, 0])

c_swing_frc = +1
c_stance_frc = 0
c_swing_spd = 0
c_stance_spd = +1
OMEGA = 4.5
RIGHT_FOOT = 13
LEFT_FOOT = 25

RIGHT_FOOT_JOINT = 15
LEFT_FOOT_JOINT = 7
PELVIS = 1

target_feet_orientation = -np.pi / 4.0
right_foot_force_idx = 49
left_foot_force_idx = 33

# The camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,  # use the body id of Cassie
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.85)),  # adjust the height to match Cassie's height
    "elevation": -20.0,
}


pos_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34]

vel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31]
sensors = OrderedDict(
    {
        "actuatorpos": [
            "left-hip-roll-input",
            "left-hip-yaw-input",
            "left-hip-pitch-input",
            "left-knee-input",
            "left-foot-input",
            "right-hip-roll-input",
            "right-hip-yaw-input",
            "right-hip-pitch-input",
            "right-knee-input",
            "right-foot-input",
        ],
        "jointpos": [
            "left-shin-output",
            "left-tarsus-output",
            "left-foot-output",
            "right-shin-output",
            "right-tarsus-output",
            "right-foot-output",
        ],
        "framequat": ["pelvis-orientation"],
        "gyro": ["pelvis-angular-velocity"],
        "accelerometer": ["pelvis-linear-acceleration"],
        "magnetometer": ["pelvis-magnetometer"],
    }
)


def create_symmetry_matrix(size, left_indices, right_indices, negated_indices=None):
    matrix = np.eye(size)
    for left, right in zip(left_indices, right_indices):
        matrix[left, right] = 1
        matrix[right, left] = 1
        matrix[left, left] = 0
        matrix[right, right] = 0

    if negated_indices:
        for idx in negated_indices:
            matrix[idx, idx] = -1

    return matrix


# Symmetry matrices for each sensor group
symmetry_matrices = {
    "actuatorpos": create_symmetry_matrix(
        10,
        left_indices=[0, 1, 2, 3, 4],
        right_indices=[5, 6, 7, 8, 9],
    ),
    "jointpos": create_symmetry_matrix(
        6, left_indices=[0, 1, 2], right_indices=[3, 4, 5]
    ),
    "framequat": create_symmetry_matrix(
        4,
        left_indices=[],
        right_indices=[],
        negated_indices=[1, 3],  # Negate z and y components of quaternion
    ),
    "gyro": create_symmetry_matrix(
        3,
        left_indices=[],
        right_indices=[],
        negated_indices=[0, 2],  # Negate y component of angular velocity
    ),
    "accelerometer": create_symmetry_matrix(
        3,
        left_indices=[],
        right_indices=[],
        negated_indices=[1],  # Negate y component of acceleration
    ),
    "magnetometer": create_symmetry_matrix(
        3,
        left_indices=[],
        right_indices=[],
        negated_indices=[1],  # Negate y component of magnetic field
    ),
    "command": create_symmetry_matrix(
        2, left_indices=[], right_indices=[], negated_indices=[1]
    ),
    "contact_forces": create_symmetry_matrix(
        6, left_indices=[0, 1, 2], right_indices=[3, 4, 5]
    ),
    "clock": create_symmetry_matrix(
        2,
        left_indices=[],
        right_indices=[],
        negated_indices=[0, 1],
    ),
}


def get_full_symmetry_matrix(symmetry_matrices):
    total_size = sum(symmetry_matrices[key].shape[0] for key in symmetry_matrices)
    full_matrix = np.zeros((total_size, total_size))

    current_index = 0
    for key in symmetry_matrices:
        size = symmetry_matrices[key].shape[0]
        symmetry_matrix = symmetry_matrices[key]
        full_matrix[
            current_index : current_index + size, current_index : current_index + size
        ] = symmetry_matrix
        current_index += size

    return full_matrix


full_symmetry_matrix = get_full_symmetry_matrix(symmetry_matrices)
