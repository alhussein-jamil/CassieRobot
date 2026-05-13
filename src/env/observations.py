from copy import deepcopy  # noqa: F401  (kept for backward-compat re-export)
from typing import TYPE_CHECKING

import numpy as np

from .constants import sensors

if TYPE_CHECKING:
    import numpy.typing as npt


def build_observation(
    sensor_data: dict[str, np.ndarray],
    command: "npt.NDArray[np.float32]",
    contact_force_left_foot: "npt.NDArray[np.float32]",
    contact_force_right_foot: "npt.NDArray[np.float32]",
    phi: float,
) -> "npt.NDArray[np.float32]":
    """Constructs the observation vector from sensor data and internal state."""
    clock_signal = np.array([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)])

    sensor_readings_np = np.concatenate(
        [sensor_data[group] for group in sensors.keys()]
    )

    return np.concatenate(
        [
            sensor_readings_np,
            command,
            contact_force_left_foot[:3],
            contact_force_right_foot[:3],
            clock_signal,
        ]
    ).astype(np.float32)


def get_symmetric_obs(obs: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
    """Returns the symmetric (mirrored) observation for alternating gait."""
    symmetric_obs = obs.copy()

    # actuatorpos - swap left and right actuators
    symmetric_obs[0:5] = obs[5:10]
    symmetric_obs[5:10] = obs[0:5]

    # Negate hip-roll and hip-yaw (sagittal-plane reflection)
    symmetric_obs[0] = -symmetric_obs[0]  # right-hip-roll -> mirrored left-hip-roll
    symmetric_obs[1] = -symmetric_obs[1]  # right-hip-yaw  -> mirrored left-hip-yaw
    symmetric_obs[5] = -symmetric_obs[5]  # left-hip-roll  -> mirrored right-hip-roll
    symmetric_obs[6] = -symmetric_obs[6]  # left-hip-yaw   -> mirrored right-hip-yaw

    # jointpos - swap left and right joints
    symmetric_obs[10:13] = obs[13:16]
    symmetric_obs[13:16] = obs[10:13]

    # pelvis quaternion symmetric along the sagittal plane (xz-plane)
    symmetric_obs[17] = -obs[17]  # x component
    symmetric_obs[19] = -obs[19]  # z component

    # pelvis angular velocity - symmetric along sagittal plane
    symmetric_obs[20] = -obs[20]  # x component
    symmetric_obs[22] = -obs[22]  # z component

    # pelvis linear acceleration - symmetric along sagittal plane
    symmetric_obs[24] = -obs[24]  # y component

    # Magnetometer - symmetric along sagittal plane
    symmetric_obs[27] = -obs[27]  # y component

    # command - y velocity flips sign when mirroring
    symmetric_obs[30] = -obs[30]  # y command

    # contact forces - swap left and right foot forces, and negate the Y-axis tangent force
    left_force = obs[34:37].copy()
    right_force = obs[31:34].copy()

    # The normal force is at index 0, tangent 1 (Y-axis) is at index 1, tangent 2 (X-axis) is at index 2
    left_force[1] = -left_force[1]
    right_force[1] = -right_force[1]

    symmetric_obs[31:34] = left_force
    symmetric_obs[34:37] = right_force

    # clock signal - phase shift of half cycle for symmetric gait
    symmetric_obs[37] = -obs[37]  # sin component
    symmetric_obs[38] = -obs[38]  # cos component

    return symmetric_obs


def symmetric_action(action: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
    """Swap left/right actuator commands for symmetric gait."""
    mirrored = np.array([*action[5:], *action[:5]])
    # Negate hip-roll and hip-yaw (sagittal-plane reflection)
    mirrored[0] = -mirrored[0]  # hip-roll
    mirrored[1] = -mirrored[1]  # hip-yaw
    mirrored[5] = -mirrored[5]  # hip-roll
    mirrored[6] = -mirrored[6]  # hip-yaw
    return mirrored
