import numpy as np
import torch

import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal

from TrackToLearn.algorithms.shared.utils import (
    format_widths, make_fc_network)
from TrackToLearn.algorithms.shared.offpolicy import Critic, DoubleCritic

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class PlanarFlow(nn.Module):
    def __init__(self, data_dim):
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.rand(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()
        self.h_prime = lambda z: (1 - self.h(z) ** 2)

    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        u = self.constrained_u()
        self.w = self.w
        self.b = self.b
        hidden_units = torch.matmul(self.w.T, z.T) + self.b
        y = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return y, log_det


class PlanarFlowTransform(nn.Module):
    def __init__(self, data_dim, num_flows):
        super(PlanarFlowTransform, self).__init__()
        self.planar_flows = nn.ModuleList(
            [PlanarFlow(data_dim) for _ in range(num_flows)]
        )

    def forward(self, x):
        y = x
        log_det_J = 0

        for flow in self.planar_flows:
            y, log_det_J_flow = flow(y)
            log_det_J += log_det_J_flow

        return y, log_det_J

    def inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        _, log_det_J = self.forward(x)
        return log_det_J


class FlowActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        num_flows: int,
    ):
        super(FlowActor, self).__init__()

        self.action_dim = action_dim
        hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(hidden_layers, state_dim, action_dim * 2)

        self.output_activation = nn.Tanh()

        self.flow_transform = PlanarFlowTransform(action_dim, num_flows)

    def forward(
        self,
        state: torch.Tensor,
        stochastic: bool,
    ) -> torch.Tensor:
        p = self.layers(state)
        mu = p[:, :self.action_dim]
        log_std = p[:, self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        det_L = log_std.prod(dim=-1)
        pi_distribution = Normal(mu, std)
        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        # Apply normalizing flow
        z, log_det_J = self.flow_transform(pi_action)

        pi_action, tan_J = self.tanh_layer(z)
        log_det_J += tan_J
        log_det_J += det_L
        return pi_action, log_det_J

    def tanh_layer(self, z):
        pi_action = self.output_activation(z)
        log_det_J = (1 - ((pi_action) ** 2)).sum(dim=-1)

        return pi_action, log_det_J


class NFActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        num_flows: int,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: int
                String representing layer widths

        """
        self.device = device
        self.actor = FlowActor(
            state_dim, action_dim, hidden_dims, num_flows,
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dims,
        ).to(device)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
        """
        return self.actor(state)

    def select_action(self, state: np.array, stochastic=False) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.act(state).cpu().data.numpy()

        return action

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.actor.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict(), self.critic.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        self.critic.load_state_dict(
            torch.load(pjoin(path, filename + '_critic.pth'),
                       map_location=self.device))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()


class NFSACActorCritic(NFActorCritic):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        num_flows: int,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        self.device = device
        self.actor = FlowActor(
            state_dim, action_dim, hidden_dims, num_flows,
        ).to(device)

        self.critic = DoubleCritic(
            state_dim, action_dim, hidden_dims,
        ).to(device)

    def act(self, state: torch.Tensor, stochastic=True) -> torch.Tensor:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
        """
        action, logprob = self.actor(state, stochastic)
        return action, logprob

    def select_action(self, state: np.array, stochastic=False) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action, _ = self.act(state, stochastic)

        return action.cpu().data.numpy()

