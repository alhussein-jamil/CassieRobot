"""Compare action magnitudes with and without the observation filter applied.

Loads the latest checkpoint, runs one short rollout under three modes, and
prints summary statistics on the produced actions and observations.

  raw     -> obs fed directly to compute_single_action (BUGGY old eval)
  filtered-> obs run through SyncedFilterAgentConnector first  (NEW eval)
  rollout -> RolloutWorker-style call via agent_connectors + policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import multiprocessing  # noqa: E402

# --- Compatibility shim: legacy checkpoints pickled `packaging.version.Version`
# with a tuple state, but modern packaging has no `__setstate__`. Provide one
# so `pickle.load` can succeed on existing checkpoints.
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
    # Highest checkpoint number per test, take the most-recent test_*
    by_test: dict[str, Path] = {}
    for p in candidates:
        by_test[p.parts[-3]] = p  # last write wins (sorted ascending)
    test = sorted(by_test.keys(), key=lambda n: int(n.split("_")[-1]))[-1]
    return by_test[test]


def stats(name: str, arr: np.ndarray) -> str:
    return (
        f"{name:24s} shape={tuple(arr.shape)} "
        f"mean={arr.mean():+.4f} std={arr.std():.4f} "
        f"min={arr.min():+.4f} max={arr.max():+.4f} "
        f"|x|_mean={np.abs(arr).mean():.4f}"
    )


def run(env: CassieEnv, policy, *, filter_conn, apply_action_conn: bool, n_steps: int):
    obs, _ = env.reset(seed=0)
    raw_obs_log, in_obs_log, raw_act_log, act_log = [], [], [], []
    for _ in range(n_steps):
        if filter_conn is not None:
            acd = AgentConnectorDataType(env_id="eval", agent_id=0, data={"obs": obs})
            obs_in = filter_conn([acd])[0].data["obs"]
        else:
            obs_in = obs
        raw_action, states, fetches = policy.compute_single_action(
            obs_in, explore=False
        )
        if apply_action_conn and getattr(policy, "action_connectors", None):
            ac = ActionConnectorDataType(
                env_id="eval",
                agent_id=0,
                input_dict={},
                output=(raw_action, states, fetches),
            )
            action = policy.action_connectors(ac).output[0]
        else:
            action = raw_action
        raw_obs_log.append(np.asarray(obs, dtype=np.float32))
        in_obs_log.append(np.asarray(obs_in, dtype=np.float32))
        raw_act_log.append(np.asarray(raw_action, dtype=np.float32))
        act_log.append(np.asarray(action, dtype=np.float32))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    return (
        np.stack(raw_obs_log),
        np.stack(in_obs_log),
        np.stack(raw_act_log),
        np.stack(act_log),
    )


def build_algo_from_config(config_path: Path):
    """Replicate src/training/manager.py build path so checkpoint restore works."""
    loader = Loader(
        logdir=ROOT / "output" / "ray_results", simdir=ROOT / "output" / "simulations"
    )
    full = loader.load_config(config_path)
    full["training"]["env_runners"]["num_env_runners"] = 0
    full["training"]["env_runners"]["num_envs_per_env_runner"] = 1
    full["training"]["env_runners"]["num_gpus_per_env_runner"] = 0
    full["training"]["resources"]["num_gpus"] = 0
    training_config = full["training"]
    return (
        PPOConfig()
        .environment(**training_config.get("environment", {}))
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(**training_config.get("env_runners", {}))
        .debugging(log_level="ERROR")
        .training(**training_config.get("training", {}))
        .framework(**training_config.get("framework", {}))
        .resources(**training_config.get("resources", {}))
        .evaluation(**training_config.get("evaluation", {}))
        .fault_tolerance(**training_config.get("fault_tolerance", {}))
        .callbacks(callbacks_class=CassieEnvCallback)
    ).build_algo()


def main() -> None:
    _ = multiprocessing  # silence unused
    ray.init(ignore_reinit_error=True, num_gpus=0, local_mode=True, log_to_driver=False)
    register_env("cassie-v0", lambda cfg: CassieEnv(cfg))

    ckpt = latest_checkpoint()
    print(f"Loading checkpoint: {ckpt}")
    algo = build_algo_from_config(ROOT / "configs" / "default_config.yaml")
    # The full algorithm_state.pkl pickle includes objects that fail to load
    # in this stripped environment ("state is not a dictionary"). Restore the
    # policy weights + filter state directly from the per-policy checkpoint
    # instead -- that's all we need for a deterministic forward pass.
    from ray.rllib.policy.policy import Policy

    restored = Policy.from_checkpoint(str(ckpt / "policies" / "default_policy"))
    weights = restored.get_weights()
    algo.get_policy().set_weights(weights)
    # Copy filter state if present
    src_filter_conn = (
        next(iter(restored.agent_connectors[SyncedFilterAgentConnector] or []), None)
        if getattr(restored, "agent_connectors", None)
        else None
    )
    policy = algo.get_policy()
    filter_conn = next(
        iter(policy.agent_connectors[SyncedFilterAgentConnector] or []), None
    )
    if src_filter_conn is not None and filter_conn is not None:
        filter_conn.filter = src_filter_conn.filter
        print("Filter state copied from checkpoint")
    print(
        f"Filter connector present: {filter_conn is not None} "
        f"({type(filter_conn).__name__ if filter_conn else 'None'})"
    )

    env = CassieEnv()
    N = 500

    print("\n=== RAW (no filter, no action-unsquash -- old buggy eval) ===")
    raw_obs, in_obs, raw_acts, acts = run(
        env, policy, filter_conn=None, apply_action_conn=False, n_steps=N
    )
    print(stats("obs (fed to policy)", in_obs))
    print(stats("action (sent to env)", acts))

    print("\n=== FILTERED only (obs filter, no action-unsquash) ===")
    raw_obs2, in_obs2, raw_acts2, acts2 = run(
        env, policy, filter_conn=filter_conn, apply_action_conn=False, n_steps=N
    )
    print(stats("obs (fed to policy)", in_obs2))
    print(stats("action (sent to env)", acts2))

    print("\n=== FILTERED + ACTION-UNSQUASHED (matches training) ===")
    raw_obs3, in_obs3, raw_acts3, acts3 = run(
        env, policy, filter_conn=filter_conn, apply_action_conn=True, n_steps=N
    )
    print(stats("obs (fed to policy)", in_obs3))
    print(stats("policy raw output (pre-unsquash)", raw_acts3))
    print(stats("action (sent to env)", acts3))

    print("\n=== Per-component action |.| mean ===")
    print("  raw eval         :", np.round(np.abs(acts).mean(0), 3))
    print("  filtered only    :", np.round(np.abs(acts2).mean(0), 3))
    print("  filtered+unsquash:", np.round(np.abs(acts3).mean(0), 3))
    print("  env action range :", np.round(env.action_space.high, 3))


if __name__ == "__main__":
    main()
