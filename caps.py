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

from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.offline import JsonReader

class TorchCustomLossModel(TorchModelV2, nn.Module):
    """PyTorch version of the CustomLossModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)

        self.sigma = model_config["caps"].get("sigma", 0.01)
        self.lambda_s = model_config["caps"].get("lambda_s", 1000)
        self.lambda_t = model_config["caps"].get("lambda_t", 1)
        # Create a new input reader per worker.
        self.fcnet = TorchFC(
            self.obs_space,
            self.action_space,
            num_outputs,
            model_config,
            name="fcnet")
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Delegate to our FCNet.
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """Calculates a custom loss on top of the given policy_loss(es).

        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs (TensorStruct): Struct of np.ndarrays holding the
                entire train batch.

        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        """
        # Get the next batch from our input files.
        batch = self.reader.next()

        # Define a secondary loss by building a graph copy with weight sharing.
        obs = restore_original_dimensions(
            torch.from_numpy(batch["obs"]).float(),
            self.obs_space,
            tensorlib="torch")
        actions = restore_original_dimensions(
            torch.from_numpy(batch["actions"]).float(),
            self.action_space,
            tensorlib="torch")
        logits, _ = self.forward({"obs": obs}, [], None)

        # get the loss from the parent class
        loss = policy_loss



        # get a bunch of normal distribution around
        dist = torch.distributions.Normal(obs, self.sigma)
        around_obs = dist.sample()

        logits_around, _ = self.forward({"obs": around_obs}, [], None)

        L_S = torch.mean(torch.mean(torch.abs(logits - logits_around), axis=1))
        L_T = torch.mean(f.action_dist(actions[1:, :], actions[:-1, :]))

        # add the loss of the state around the observations to the loss
        loss += self.lambda_s * L_S
        loss += self.lambda_t * L_T
        return loss
        # # You can also add self-supervised losses easily by referencing tensors
        # # created during _build_layers_v2(). For example, an autoencoder-style
        # # loss can be added as follows:
        # # ae_loss = squared_diff(
        # #     loss_inputs["obs"], Decoder(self.fcnet.last_layer))
        # print("FYI: You can also use these tensors: {}, ".format(loss_inputs))

        # # Compute the IL loss.
        # action_dist = TorchCategorical(logits, self.model_config)
        # self.policy_loss = policy_loss
        # self.imitation_loss = torch.mean(
        #     -action_dist.logp(torch.from_numpy(batch["actions"])))

        # # Add the imitation loss to each already calculated policy loss term.
        # # Alternatively (if custom loss has its own optimizer):
        # # return policy_loss + [10 * self.imitation_loss]
        # return [l + 10 * self.imitation_loss for l in policy_loss]

    # def custom_stats(self):
    #     return {
    #         "policy_loss": torch.mean(self.policy_loss),
    #         "imitation_loss": self.imitation_loss,
    #     }