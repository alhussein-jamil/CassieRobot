from typing import List, Type, Union

import torch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType

import functions as f


class CapsTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.sigma = config["caps"].get("sigma", 0.01)
        self.lambda_s = config["caps"].get("lambda_s", 1000)
        self.lambda_t = config["caps"].get("lambda_t", 1)
        super().__init__(observation_space, action_space, config)

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        # get the loss from the parent class
        loss = super().loss(model, dist_class, train_batch)

        # get the observations and actions
        obs, actions = train_batch["obs"], train_batch["actions"]

        # get the logits and the state of the model
        logits, _ = model({"obs": obs})

        # get a bunch of normal distribution around
        dist = torch.distributions.Normal(obs, self.sigma)

        around_obs = dist.sample()

        logits_around, _ = model({"obs": around_obs})

        L_S = torch.mean(torch.mean(torch.abs(logits - logits_around), axis=1))
        L_T = torch.mean(f.action_dist(actions[1:, :], actions[:-1, :]))

        # add the loss of the state around the observations to the loss
        loss += self.lambda_s * L_S
        loss += self.lambda_t * L_T

        return loss
