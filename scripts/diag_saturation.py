"""Diagnose why the trained Cassie policy saturates actuators.

Checks two suspects:
  1) Raw policy mean is unbounded (no tanh / squashing on the output head),
     pushing every component into the unsquash → clip region of MuJoCo.
  2) The reward's ``q_torque`` normalization range is too small, so the
     torque penalty saturates to a constant (no gradient pressure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# packaging compat shim (legacy checkpoints pickle Version with a tuple state)
from packaging.version import Version  # noqa: E402


def _version_setstate(self, state):
    if isinstance(state, dict):
        self.__dict__.update(state)
        return
    if isinstance(state, tuple):
        epoch, release = state[0], state[1]
        vstr = ".".join(str(x) for x in release)
        if epoch:
            vstr = f"{epoch}!{vstr}"
        self.__dict__.update(Version(vstr).__dict__)
        return
    raise TypeError(f"Unsupported Version state: {type(state)}")


Version.__setstate__ = _version_setstate

import ray  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.rllib.connectors.agent.synced_filter import (  # noqa: E402
    SyncedFilterAgentConnector,
)
from ray.rllib.policy.policy import Policy  # noqa: E402
from ray.rllib.utils.typing import (  # noqa: E402
    ActionConnectorDataType,
    AgentConnectorDataType,
)
from ray.tune.registry import register_env  # noqa: E402

from src.env.callbacks import CassieEnvCallback  # noqa: E402
from src.env.cassie import CassieEnv  # noqa: E402
from src.training.loader import Loader  # noqa: E402


def latest_checkpoint() -> Path:
    candidates = sorted(
        (ROOT / "output" / "checkpoints").glob("test_*/config_0/checkpoint_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    by_test: dict[str, Path] = {}
    for p in candidates:
        by_test[p.parts[-3]] = p
    test = sorted(by_test.keys(), key=lambda n: int(n.split("_")[-1]))[-1]
    return by_test[test]


def build_algo(config_path: Path):
    loader = Loader(
        logdir=ROOT / "output" / "ray_results",
        simdir=ROOT / "output" / "simulations",
    )
    full = loader.load_config(config_path)
    full["training"]["env_runners"]["num_env_runners"] = 0
    full["training"]["env_runners"]["num_envs_per_env_runner"] = 1
    full["training"]["env_runners"]["num_gpus_per_env_runner"] = 0
    full["training"]["resources"]["num_gpus"] = 0
    tc = full["training"]
    return (
        PPOConfig()
        .environment(**tc.get("environment", {}))
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(**tc.get("env_runners", {}))
        .debugging(log_level="ERROR")
        .training(**tc.get("training", {}))
        .framework(**tc.get("framework", {}))
        .resources(**tc.get("resources", {}))
        .evaluation(**tc.get("evaluation", {}))
        .fault_tolerance(**tc.get("fault_tolerance", {}))
        .callbacks(callbacks_class=CassieEnvCallback)
    ).build_algo()


def main():
    ray.init(ignore_reinit_error=True, num_gpus=0, local_mode=True, log_to_driver=False)
    register_env("cassie-v0", lambda cfg: CassieEnv(cfg))

    ckpt = latest_checkpoint()
    print(f"Checkpoint: {ckpt}\n")
    algo = build_algo(ROOT / "configs" / "default_config.yaml")
    restored = Policy.from_checkpoint(str(ckpt / "policies" / "default_policy"))
    algo.get_policy().set_weights(restored.get_weights())
    src_filter = next(
        iter(restored.agent_connectors[SyncedFilterAgentConnector] or []), None
    )
    policy = algo.get_policy()
    filter_conn = next(
        iter(policy.agent_connectors[SyncedFilterAgentConnector] or []), None
    )
    if src_filter is not None and filter_conn is not None:
        filter_conn.filter = src_filter.filter

    # --- Inspect the policy's free log_std parameter ---
    print("=== Policy parameters ===")
    weights = policy.get_weights()
    for k, v in weights.items():
        print(f"  {k:50s} shape={v.shape}  |x|_mean={np.abs(v).mean():.4f}")
        if "log_std" in k.lower():
            print(f"     -> std = exp(log_std) = {np.exp(v)}")

    # --- Run rollout, log raw policy mean + sampled action ---
    env = CassieEnv()
    obs, _ = env.reset(seed=0)
    raw_outputs, sampled_actions, env_actions, q_torque_log = [], [], [], []
    for _ in range(500):
        acd = AgentConnectorDataType(env_id="eval", agent_id=0, data={"obs": obs})
        obs_in = filter_conn([acd])[0].data["obs"]
        # Deterministic (mean) output
        raw_a, states, fetches = policy.compute_single_action(obs_in, explore=False)
        raw_outputs.append(np.asarray(raw_a, dtype=np.float32))
        # Stochastic sample, for entropy diagnosis
        s_a, _, _ = policy.compute_single_action(obs_in, explore=True)
        sampled_actions.append(np.asarray(s_a, dtype=np.float32))
        # Apply action-connector pipeline (matches training)
        ac = ActionConnectorDataType(
            env_id="eval",
            agent_id=0,
            input_dict={},
            output=(raw_a, states, fetches),
        )
        env_a = policy.action_connectors(ac).output[0]
        env_actions.append(np.asarray(env_a, dtype=np.float32))
        q_torque_log.append(float(np.linalg.norm(env_a)))
        obs, _, term, trunc, _ = env.step(env_a)
        if term or trunc:
            obs, _ = env.reset()

    raw_outputs = np.stack(raw_outputs)
    sampled_actions = np.stack(sampled_actions)
    env_actions = np.stack(env_actions)
    q_torque_log = np.array(q_torque_log)

    print("\n=== Raw policy mean output (pre-unsquash, pre-clip) ===")
    print(f"  mean= {raw_outputs.mean(0).round(3)}")
    print(f"  std=  {raw_outputs.std(0).round(3)}")
    print(f"  |x|_mean= {np.abs(raw_outputs).mean(0).round(3)}")
    print(
        f"  fraction with |a|>1 per component (i.e. would saturate): "
        f"{(np.abs(raw_outputs) > 1).mean(0).round(3)}"
    )
    print(f"  fraction with |a|>5: {(np.abs(raw_outputs) > 5).mean(0).round(3)}")

    print("\n=== Stochastic samples (explore=True) ===")
    print(f"  std per component: {sampled_actions.std(0).round(3)}")

    print("\n=== Env action saturation ===")
    high = env.action_space.high
    sat = np.isclose(np.abs(env_actions), high[None, :], atol=1e-3).mean(0)
    print(f"  fraction at limit per component: {sat.round(3)}")
    print(
        f"  per-component |a|/limit ratio:   {(np.abs(env_actions).mean(0) / high).round(3)}"
    )

    print("\n=== q_torque normalization sanity check ===")
    actual_max_norm = np.linalg.norm(high)
    print(
        f"  Configured normalization upper bound: max(action_space.high) = {high.max():.2f}"
    )
    print(
        f"  Actual max possible ||action||:        ||action_space.high|| = {actual_max_norm:.2f}"
    )
    print(
        f"  Observed ||action|| stats: mean={q_torque_log.mean():.2f} "
        f"max={q_torque_log.max():.2f} "
        f"fraction>=12.2 (saturates penalty): {(q_torque_log >= 12.2).mean():.2f}"
    )

    # === Rapidness / jerk ===
    # Step-to-step change in env action, normalized by the actuator range so
    # 1.0 = "swung from -limit to +limit in one control step".
    # Cassie control timestep: timestep * frame_skip = 0.0005 * 60 = 0.03 s
    # (or whatever is configured); per-second slew = delta / dt.
    diffs = np.diff(env_actions, axis=0)
    rng = (env.action_space.high - env.action_space.low)[None, :]
    norm_diffs = np.abs(diffs) / rng
    dt = float(getattr(env, "secs_per_env_step", 0.03))
    print("\n=== Rapidness (action change per control step) ===")
    print(f"  control dt = {dt * 1000:.1f} ms")
    print(f"  |Δa|/range mean per component: {norm_diffs.mean(0).round(3)}")
    print(f"  |Δa|/range max  per component: {norm_diffs.max(0).round(3)}")
    print(
        f"  fraction of steps with |Δa|/range > 0.5 (swing >half range): "
        f"{(norm_diffs > 0.5).mean(0).round(3)}"
    )
    print(
        f"  fraction of steps with |Δa|/range > 0.9 (swing >90% range): "
        f"{(norm_diffs > 0.9).mean(0).round(3)}"
    )
    slew = np.abs(diffs) / dt
    print(f"  mean slew rate (Nm/s or rad/s) per component: {slew.mean(0).round(2)}")
    print(f"  max  slew rate per component:                 {slew.max(0).round(2)}")

    # Sign-flip rate per component: how often sign(a_t) != sign(a_{t-1})
    sgn = np.sign(env_actions)
    sign_flips = (sgn[1:] * sgn[:-1] < 0).mean(0)
    print(
        f"  sign-flip rate per component (bang-bang indicator): {sign_flips.round(3)}"
    )
    print(
        "  -> if these are >0.3 the policy is essentially bang-banging the actuators."
    )


if __name__ == "__main__":
    main()
