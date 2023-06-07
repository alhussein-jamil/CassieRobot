import torch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from typing import Type, Union, List, Any, Dict
import functions as f
import constants as c
from ray.rllib.algorithms.ppo import PPO

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
class CapsModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                              name)
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        print("caps loss")
        # get the loss from the parent class
        loss = super().loss(model, dist_class, train_batch)
        print("loss", loss)
        # get the observations and actions
        obs, actions = train_batch["obs"], train_batch["actions"]

        # get the logits and the state of the model
        logits, _ = model({"obs": obs})

        # get a bunch of normal distribution around
        dist = torch.distributions.Normal(obs, c.caps_sigma)
        around_obs = dist.sample()

        logits_around, _ = model({"obs": around_obs})

        L_S = torch.mean(torch.mean(torch.abs(logits - logits_around), axis=1))
        L_T = torch.mean(f.action_dist(actions[1:, :], actions[:-1, :]))

        # add the loss of the state around the observations to the loss
        loss += c.caps_lambda_s * L_S
        loss += c.caps_lambda_t * L_T
        return loss
