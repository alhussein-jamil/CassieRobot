import logging as log
from typing import Any, Dict, TYPE_CHECKING

import numba as nb
import numpy as np

from .constants import (
    FORWARD_QUARTERNIONS,
    OMEGA,
    THETA_LEFT,
    THETA_RIGHT,
    c_stance_spd,
    c_swing_frc,
    gravity,
    mass,
)
from .functions import action_dist

if TYPE_CHECKING:
    import numpy.typing as npt


class RewardCalculator:
    """Handles all reward computation for the Cassie environment."""

    def __init__(
        self,
        reward_coeffs: Dict[str, float],
        x_cmd_vel: float,
        y_cmd_vel: float,
        action_space_high: "npt.NDArray[np.float32]",
        action_space_low: "npt.NDArray[np.float32]",
        steps_per_cycle: int,
        von_mises_values_swing: "npt.NDArray[np.float32]",
        von_mises_values_stance: "npt.NDArray[np.float32]",
    ):
        self.reward_coeffs = reward_coeffs
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low
        self.steps_per_cycle = steps_per_cycle
        self.von_mises_values_swing = von_mises_values_swing
        self.von_mises_values_stance = von_mises_values_stance

        # Fixed normalization ranges for reward components
        # Force: body weight (gravity * mass) gives a smooth gradient from
        #   no-contact (exp=1) through light contact (~0.5) to full stance (~0).
        # Speed: use actuator speed limits rather than command velocity so
        #   joint velocities don't saturate the range.
        max_joint_spd = 12.2  # rad/s, from actuator speed limits (knee)
        # Use ||action_space.high|| as the upper bound for ``q_torque``
        # (= ||action||) and sqrt(n_actions) as the upper bound for the
        # per-action change ``q_action`` (which is normalized to [0, 1] per
        # component before being summed in quadrature). The previous bounds
        # (max(action_space.high) and 1.0) were too small, so both quantities
        # saturated at every step and produced ZERO gradient on torque magnitude
        # / action smoothness.
        torque_max_norm = float(np.linalg.norm(action_space_high))
        action_dist_max = float(np.sqrt(action_space_high.size))
        self.exponents_ranges = {
            "q_vx": (0.0, max(x_cmd_vel, 0.5)),
            "q_vy": (0.0, max(y_cmd_vel, 0.5)),
            "q_left_frc": (0.0, gravity * mass),
            "q_right_frc": (0.0, gravity * mass),
            "q_left_spd": (0.0, max_joint_spd),
            "q_right_spd": (0.0, max_joint_spd),
            "q_action": (0.0, action_dist_max),
            "q_pelvis_acc": (0.0, 10.0),
            "q_orientation": (0.0, 2 * np.sqrt(2)),
            "q_torque": (0.0, torque_max_norm),
            "q_left_foot_pitch": (0.0, np.pi / 2),
            "q_right_foot_pitch": (0.0, np.pi / 2),
        }

    def _normalize_quantity(self, name: str, q: float) -> float:
        """Normalizes a quantity using fixed ranges."""
        lo, hi = self.exponents_ranges[name]
        rng = hi - lo + 1e-6
        return np.clip((q - lo) / rng, 0.0, 1.0)

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def _weighted_sum(
        rewards: "npt.NDArray[np.float32]", coeffs: "npt.NDArray[np.float32]"
    ) -> float:
        return np.sum(rewards * coeffs) / np.sum(coeffs)

    def normalize_reward(
        self, rewards: Dict[str, float], reward_coeffs: Dict[str, float]
    ) -> float:
        """Normalizes reward components based on their coefficients."""
        reward_keys = sorted(rewards.keys())
        reward_values = np.array([rewards[k] for k in reward_keys], dtype=np.float32)
        coeffs_values = np.array(
            [reward_coeffs[k] for k in reward_keys if k in reward_coeffs],
            dtype=np.float32,
        )

        if len(reward_values) != len(coeffs_values):
            log.warning(
                "Mismatch between reward components (%d) and coefficients (%d)",
                len(reward_values),
                len(coeffs_values),
            )
            matched_keys = [k for k in reward_keys if k in reward_coeffs]
            reward_values = np.array(
                [rewards[k] for k in matched_keys], dtype=np.float32
            )
            coeffs_values = np.array(
                [reward_coeffs[k] for k in matched_keys], dtype=np.float32
            )

        if np.sum(coeffs_values) == 0:
            log.warning("Sum of reward coefficients is zero, returning raw sum.")
            return np.sum(reward_values)

        return self._weighted_sum(reward_values, coeffs_values)

    def _phase_coefficient_force(self, phi: float) -> float:
        """Coefficient for force reward based on phase."""
        idx = int(round((phi % 1.0) * self.steps_per_cycle)) % self.steps_per_cycle
        return c_swing_frc * self.von_mises_values_swing[idx]

    def _phase_coefficient_speed(self, phi: float) -> float:
        """Coefficient for speed reward based on phase."""
        idx = int(round((phi % 1.0) * self.steps_per_cycle)) % self.steps_per_cycle
        return c_stance_spd * self.von_mises_values_stance[idx]

    def compute(
        self,
        action: "npt.NDArray[np.float32]",
        previous_action: "npt.NDArray[np.float32]",
        qvel: "npt.NDArray[np.float64]",
        command: "npt.NDArray[np.float32]",
        pelvis_quat: "npt.NDArray[np.float32]",
        pelvis_ang_vel: "npt.NDArray[np.float32]",
        left_foot_rpy: "npt.NDArray[np.float32]",
        right_foot_rpy: "npt.NDArray[np.float32]",
        contact_force_left: "npt.NDArray[np.float32]",
        contact_force_right: "npt.NDArray[np.float32]",
        phi: float,
    ) -> tuple[float, Dict[str, Any]]:
        """Computes all reward components and the final reward value."""

        # --- Raw quantities ---
        raw_quantities = {
            "q_vx": abs(qvel[0] - command[0]),
            "q_vy": abs(qvel[1] - command[1]),
            "q_left_frc": np.linalg.norm(contact_force_left),
            "q_right_frc": np.linalg.norm(contact_force_right),
            "q_left_spd": abs(qvel[12]),
            "q_right_spd": abs(qvel[19]),
            "q_action": action_dist(
                action.reshape(1, -1),
                previous_action.reshape(1, -1),
                self.action_space_high,
                self.action_space_low,
            )[0],
            "q_pelvis_acc": np.linalg.norm(pelvis_ang_vel),
            "q_torque": np.linalg.norm(action),
            "q_orientation": np.linalg.norm(pelvis_quat - FORWARD_QUARTERNIONS),
            "q_left_foot_pitch": (
                abs(left_foot_rpy[1])
                if np.linalg.norm(contact_force_left) > 0.01
                else 0.0
            ),
            "q_right_foot_pitch": (
                abs(right_foot_rpy[1])
                if np.linalg.norm(contact_force_right) > 0.01
                else 0.0
            ),
        }

        # --- Normalize and apply exponential ---
        normalized_quantities = {
            k: self._normalize_quantity(k, v) for k, v in raw_quantities.items()
        }
        after_exponential = {
            k: np.exp(-OMEGA * v) for k, v in normalized_quantities.items()
        }

        # --- Combine reward components ---
        c_frc = self._phase_coefficient_force
        c_spd = self._phase_coefficient_speed

        # Command following
        r_cmd = (
            1.0 * after_exponential["q_vx"]
            + 0.8 * after_exponential["q_vy"]
            + 0.2 * after_exponential["q_orientation"]
        ) / (1.0 + 0.8 + 0.2)

        # Smoothness
        r_smooth = (
            1.0 * after_exponential["q_action"]
            + 0.5 * after_exponential["q_torque"]
            + 0.5 * after_exponential["q_pelvis_acc"]
        ) / (1.0 + 0.5 + 0.5)

        # Bipedal locomotion (phase-modulated)
        r_biped = (
            c_frc(phi + THETA_LEFT) * after_exponential["q_left_frc"]
            + c_frc(phi + THETA_RIGHT) * after_exponential["q_right_frc"]
            + c_spd(phi + THETA_LEFT) * after_exponential["q_left_spd"]
            + c_spd(phi + THETA_RIGHT) * after_exponential["q_right_spd"]
        ) / 2.0

        # Feet parallel to ground
        r_feet_parallel = (
            after_exponential["q_left_foot_pitch"]
            + after_exponential["q_right_foot_pitch"]
        ) / 2.0

        rewards: Dict[str, float] = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth,
            "r_feet_parallel": r_feet_parallel,
        }

        total_reward = self.normalize_reward(rewards, self.reward_coeffs)
        total_reward += self.reward_coeffs.get("r_bias", 0.0)

        coeff = {
            "c_frc_left": c_frc(phi + THETA_LEFT),
            "c_frc_right": c_frc(phi + THETA_RIGHT),
            "c_spd_left": c_spd(phi + THETA_LEFT),
            "c_spd_right": c_spd(phi + THETA_RIGHT),
        }

        metrics = {
            "raw_quantities": raw_quantities,
            "normalized_quantities": normalized_quantities,
            "after_exponential": after_exponential,
            "coefficients": coeff,
            "rewards": rewards,
            "total_reward": total_reward,
        }

        return total_reward, metrics
