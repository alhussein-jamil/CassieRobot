import numpy as np


DEFAULT_CONFIG = {
    "symmetric_regulation": True,
    "steps_per_cycle": 30,
    "r": 1.0,
    "kappa": 25,
    "x_cmd_vel": 1.5,
    "y_cmd_vel": 0,
    "z_cmd_vel": 0,
    "terminate_when_unhealthy": True,
    "max_simulation_steps": 400,
    "pelvis_height": [0.75, 1.25],
    "feet_distance_x": 1.0,
    "feet_distance_y": 0.5,
    "feet_distance_z": 0.5,
    "feet_pelvis_height": 0.3,
    "feet_height": 0.6,
    "model": "cassie",
    "render_mode": "rgb_array",
    "reset_noise_scale": 0.01,
    "force_max_norm": 0.0,
    "push_freq": 0,
    "push_duration": 0,
    "bias": 1.0,
    "r_biped": 4.0,
    "r_cmd": 3.0,
    "r_smooth": 1.0,
    "r_alternate": 4.0,
    "r_symmetric": 2.0,
    "is_training": True,
    "max_roll": 0.2,
    "max_pitch": 0.2,
    "max_yaw": 0.2,
}


# The constants are defined here
THETA_LEFT = 0.5
THETA_RIGHT = 0

FORWARD_QUARTERNIONS = np.array([1, 0, 0, 0])

c_swing_frc = +1
c_stance_frc = 0
c_swing_spd = 0
c_stance_spd = +1


OMEGA = 4.5


# Data indices
PELVIS = 1

RIGHT_FOOT = 13
LEFT_FOOT = 25

RIGHT_CONTACT_IDX = 49
LEFT_CONTACT_IDX = 33

# The camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,  # use the body id of Cassie
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.85)),  # adjust the height to match Cassie's height
    "elevation": -20.0,
}

actuator_speed_ranges = {
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
    "q_vx": [0, 1.0],
    "q_vy": [0, 0.5],
    "q_vz": [0, 5],
    "q_frc": [0, 1e6],
    "q_spd": [0, 2],
    "q_action": [0, 3],
    "q_orientation": [0, 1],
    "q_torque": [0, 30],
    "q_pelvis_acc": [0, 100],
    "q_marche_distance": [-1.0, 1.0],
    "q_feet_orientation": [-np.pi, np.pi],
    "q_symmetric": [0.0, 5.0],
}
multiplicators = {}
for key in exponential_bornes.keys():
    multiplicators[key] = -OMEGA / (
        exponential_bornes[key][1] - exponential_bornes[key][0]
    )
