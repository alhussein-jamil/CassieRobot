from pathlib import Path

import mediapy as media
from gymnasium import Env
from loguru import logger
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.connectors.agent.synced_filter import SyncedFilterAgentConnector
from ray.rllib.utils.typing import (
    ActionConnectorDataType,
    AgentConnectorDataType,
)


class Evaluator:
    """Handles model evaluation and video rendering."""

    def __init__(self, run_config: dict):
        self.render_fps = run_config.get("render_fps", 30)

    @staticmethod
    def _get_filter_connector(policy):
        """Return the policy's observation-filter agent connector, if any.

        ``Policy.compute_single_action`` does not run the agent-connector
        pipeline, so when an ``observation_filter`` (e.g. ``MeanStdFilter``) is
        configured on the algorithm, raw observations would be fed to a policy
        that was trained on normalized ones. Pulling the connector out lets us
        apply it manually in the eval loop.
        """
        if not getattr(policy, "agent_connectors", None):
            return None
        connectors = policy.agent_connectors[SyncedFilterAgentConnector]
        return connectors[0] if connectors else None

    @staticmethod
    def _apply_filter(filter_conn, obs):
        """Run ``obs`` through the given filter connector (no-op if None)."""
        if filter_conn is None:
            return obs
        acd = AgentConnectorDataType(env_id="eval", agent_id=0, data={"obs": obs})
        return filter_conn([acd])[0].data["obs"]

    @staticmethod
    def _apply_action_connectors(policy, raw_action, states, fetches):
        """Run the policy's full action-connector pipeline (no-op if absent).

        ``Policy.compute_single_action`` skips action connectors, so when
        ``normalize_actions=True`` / ``clip_actions=True`` are configured,
        the policy's output (in [-1, 1] for normalize) would be fed straight
        to the env which expects the actuator's true range (e.g. [-12.2, 12.2]
        for the Cassie knee). Apply the connectors here to mirror what the
        rollout worker does at training time.
        """
        action_conns = getattr(policy, "action_connectors", None)
        if not action_conns:
            return raw_action
        ac_data = ActionConnectorDataType(
            env_id="eval",
            agent_id=0,
            input_dict={},
            output=(raw_action, states, fetches),
        )
        out = action_conns(ac_data)
        return out.output[0]

    def evaluate(
        self,
        trainer: PPO,
        env: Env,
        epoch: int,
        config_index: int,
        test_dir: Path,
        sim_fps: int,
    ) -> None:
        """Run one evaluation episode and save a video."""
        env.reset()
        obs, _ = env.reset()
        done = False
        frames = []
        steps = 0
        render_step = 0

        policy = trainer.get_policy()
        filter_conn = self._get_filter_connector(policy)
        if filter_conn is not None:
            logger.info(
                "Eval: applying observation filter {}", type(filter_conn).__name__
            )

        render_fps = self.render_fps
        if render_fps <= 0 or sim_fps <= 0:
            logger.warning("render_fps or sim_fps is non-positive. Skipping rendering.")
            render_fps = 0

        while not done:
            obs_in = self._apply_filter(filter_conn, obs)
            raw_action, states, fetches = policy.compute_single_action(
                obs_in, explore=False
            )
            action = self._apply_action_connectors(policy, raw_action, states, fetches)

            obs, _, done, _, _ = env.step(action)

            should_render = False
            if render_fps > 0 and steps * render_fps >= render_step * sim_fps:
                should_render = True
                render_step += 1

            if should_render:
                frame = env.render()
                frames.append(frame)

            steps += 1

        logger.info("Episode finished after {} steps", steps)
        logger.info("Episode done reason: {}", env.isdone)

        video_path = test_dir / f"config_{config_index}/run_{epoch}.mp4"
        media.write_video(video_path, frames, fps=min(render_fps, sim_fps))
        logger.info("Test saved at {}", video_path)
