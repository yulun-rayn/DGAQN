import logging

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import torch_geometric as pyg

from gnn_embed import sGAT

from utils.graph_utils import get_batch_shift

#####################################################
#                 BATCHED OPERATIONS                #
#####################################################

def batched_argmax(values, batch, eps=5e-2):
    count = torch.bincount(batch)
    count = torch.index_select(count, 0, batch)

    value_max = pyg.nn.global_max_pool(values, batch)
    value_max = torch.index_select(value_max, 0, batch)

    probs = (values == value_max).type(values.dtype)
    probs += eps / (count.type(values.dtype) - 1)
    probs[torch.isinf(probs)] = 1
    probs[torch.isnan(probs)] = 1
    return probs

def batched_sample(probs, batch):
    unique = torch.flip(torch.unique(batch.cpu(), sorted=False).to(batch.device),
                        dims=(0,)) # temp fix due to torch.unique bug
    mask = batch.unsqueeze(0) == unique.unsqueeze(1)

    p = probs * mask
    m = Categorical(p)
    a = m.sample()
    return a

#####################################################
#                       GAQN                        #
#####################################################

class TargetGAQN(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 gamma,
                 eps_greed,
                 double_q,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden):
        super(TargetGAQN, self).__init__()
        self.double_q = double_q

        # actor
        self.actor = GAQN_Actor(eps_greed)
        # critic
        self.critic = GAQN_Critic(gamma,
                                  input_dim,
                                  nb_edge_types,
                                  use_3d,
                                  gnn_nb_layers,
                                  gnn_nb_hidden,
                                  enc_nb_layers,
                                  enc_nb_hidden)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas, eps=eps)

        self.critic_target = GAQN_Critic(gamma,
                                            input_dim,
                                            nb_edge_types,
                                            use_3d,
                                            gnn_nb_layers,
                                            gnn_nb_hidden,
                                            enc_nb_layers,
                                            enc_nb_hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def forward(self):
        raise NotImplementedError

    def select_action(self, candidates, batch_idx):
        Qs = self.critic.get_value(candidates)
        actions = self.actor.select_action(Qs, batch_idx, return_shifted=False)

        return actions.squeeze_().tolist()

    def select_value(self, candidates, batch_idx):
        Qs = self.critic_target.get_value(candidates)
        shifted_actions = self.actor.select_action(Qs, batch_idx, eps=0)

        return Qs[shifted_actions], Qs

    def update(self, states, candidates, rewards, discounts, old_values, old_Qs, batch_idx):
        if self.double_q:
            Qs = self.critic.get_value(candidates)
            shifted_actions = self.actor.select_action(Qs, batch_idx, eps=0)
            old_values = old_Qs[shifted_actions]
        loss = self.critic.loss(states, rewards, old_values, discounts)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())


class GAQN_Critic(nn.Module):
    def __init__(self,
                 gamma,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden):
        super(GAQN_Critic, self).__init__()
        self.gamma = gamma

        if not isinstance(gnn_nb_hidden, list):
            gnn_nb_hidden = [gnn_nb_hidden] * gnn_nb_layers
        else:
            assert len(gnn_nb_hidden) == gnn_nb_layers
        if not isinstance(enc_nb_hidden, list):
            enc_nb_hidden = [enc_nb_hidden] * enc_nb_layers
        else:
            assert len(enc_nb_hidden) == enc_nb_layers

        # gnn encoder
        self.gnn = sGAT(input_dim, nb_edge_types, gnn_nb_hidden, gnn_nb_layers, use_3d=use_3d)
        if gnn_nb_layers == 0:
            in_dim = input_dim
        else:
            in_dim = gnn_nb_hidden[-1]

        # mlp encoder
        layers = []
        for i in range(enc_nb_layers):
            layers.append(nn.Linear(in_dim, enc_nb_hidden[i]))
            in_dim = enc_nb_hidden[i]

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(in_dim, 1)
        self.act = nn.ReLU()

        self.MseLoss = nn.MSELoss()

    def forward(self, candidates):
        X = self.gnn.get_embedding(candidates, detach=False)
        for l in self.layers:
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)

    def get_value(self, candidates):
        with torch.autograd.no_grad():
            values = self(candidates)
        return values.detach()

    def loss(self, states, rewards, old_values, discounts):
        values = self(states)
        targets = rewards + discounts * old_values
        loss = self.MseLoss(values, targets)

        return loss


class GAQN_Actor(nn.Module):
    def __init__(self, eps_greed):
        super(GAQN_Actor, self).__init__()
        self.eps_greed = eps_greed

    def forward(self):
        raise NotImplementedError

    def select_action(self, values, batch_idx, eps=None, return_shifted=True):
        if eps is None:
            eps = self.eps_greed
        probs = batched_argmax(values, batch_idx, eps=eps)
        shifted_actions = batched_sample(probs, batch_idx)

        if return_shifted:
            return shifted_actions
        else:
            return shifted_actions - get_batch_shift(batch_idx)
