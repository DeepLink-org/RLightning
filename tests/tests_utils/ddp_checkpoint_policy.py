import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from rlightning.policy.base_policy import BasePolicy
from rlightning.utils.registry import POLICIES


@POLICIES.register("DDPCheckpointPolicy")
class DDPCheckpointPolicy(BasePolicy):
    def construct_network(self, env_meta=None, *args, **kwargs):
        self.linear = nn.Linear(4, 2, bias=True)
        nn.init.constant_(self.linear.weight, 0.1)
        nn.init.constant_(self.linear.bias, 0.0)
        if torch.cuda.is_available():
            self.linear.cuda()

    def setup_optimizer(self, optim_cfg):
        lr = getattr(optim_cfg, "lr", 1e-2)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr)

    def update_dataset(self, data):
        self._dataset = data

    def train(self):
        x = self._dataset["x"].cuda()
        y = self._dataset["y"].cuda()
        pred = self.linear(x)
        loss = F.mse_loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def rollout_step(self, env_ret):
        raise NotImplementedError("DDP checkpoint test policy does not support rollout.")

    def postprocess(self, data):
        raise NotImplementedError("DDP checkpoint test policy does not support postprocess.")

    def get_trainable_parameters(self):
        state_dict = {}
        for name, model in self.model_list:
            module = model.module if isinstance(model, DDP) else model
            state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        for name, model in self.model_list:
            model.load_state_dict(state_dict[name], strict=True)
