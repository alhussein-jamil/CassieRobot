from typing import TYPE_CHECKING, Tuple

import numpy as np

from .constants import (
    PELVIS,
    LEFT_FOOT,
    RIGHT_FOOT,
)
from .functions import mod

if TYPE_CHECKING:
    import numpy.typing as npt


def check_health(
    data,
    pelvis_z_range: Tuple[float, float],
    feet_distance_x: float,
    feet_distance_y: float,
    feet_distance_z: float,
    dis_to_pelvis: float,
    feet_height: float,
    max_roll: float,
    max_pitch: float,
    max_yaw: float,
    total_sim_time: float,
    max_sim_time: int,
    contact: bool,
    foot_in_contact: bool,
    init_rpy: "npt.NDArray[np.float32]",
    pelvis_rpy: "npt.NDArray[np.float32]",
) -> Tuple[bool, str, float]:
    """
    Check if the robot is in a healthy state.

    Returns:
        (is_healthy, reason_string, done_code)
    """
    min_z, max_z = pelvis_z_range

    if (contact and data.xpos[PELVIS, 2] > max_z) or data.xpos[PELVIS, 2] < min_z:
        return False, f"Pelvis not in range: {data.xpos[PELVIS, 2]}", 1.0

    if total_sim_time > max_sim_time:
        return False, "Max steps reached", 2.0

    dx = abs(data.xpos[LEFT_FOOT, 0] - data.xpos[RIGHT_FOOT, 0])
    if not dx < feet_distance_x:
        return False, f"Feet distance out of range along x-axis: {dx}", 3.0

    dy = data.xpos[RIGHT_FOOT, 1] - data.xpos[LEFT_FOOT, 1]
    if not (0.0 < dy < feet_distance_y):
        return False, f"Feet distance out of range along y-axis: {dy}", 4.0

    dz = abs(data.xpos[LEFT_FOOT, 2] - data.xpos[RIGHT_FOOT, 2])
    if not dz < feet_distance_z:
        return False, f"Feet distance out of range along z-axis: {dz}", 5.0

    if contact and not foot_in_contact:
        return False, "Both Feet not on the ground", 6.0

    left_pelvis_dist = data.xpos[PELVIS, 2] - data.xpos[LEFT_FOOT, 2]
    if not dis_to_pelvis < left_pelvis_dist:
        return False, f"Left foot too close to pelvis: {left_pelvis_dist}", 7.0

    right_pelvis_dist = data.xpos[PELVIS, 2] - data.xpos[RIGHT_FOOT, 2]
    if not dis_to_pelvis < right_pelvis_dist:
        return False, f"Right foot too close to pelvis: {right_pelvis_dist}", 8.0

    rpy_diff = np.abs(mod(pelvis_rpy - init_rpy, np.pi))
    if rpy_diff[0] > max_roll:
        return False, f"Roll too high: {rpy_diff[0]}", 9.0
    if rpy_diff[1] > max_pitch:
        return False, f"Pitch too high: {rpy_diff[1]}", 10.0
    if rpy_diff[2] > max_yaw:
        return False, f"Yaw too high: {rpy_diff[2]}", 11.0

    return True, "not done", 0.0
