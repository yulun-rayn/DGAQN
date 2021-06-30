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
                 eps_clip,
                 double_q,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(TargetGAQN, self).__init__()
        self.double_q = double_q

        # actor
        self.actor = GAQN_Actor(eps_clip)
        # critic
        self.critic = GAQN_Critic(gamma,
                                  input_dim,
                                  nb_edge_types,
                                  use_3d,
                                  gnn_nb_layers,
                                  gnn_nb_hidden,
                                  mlp_nb_layers,
                                  mlp_nb_hidden)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas, eps=eps)

        self.critic_target = GAQN_Critic(gamma,
                                            input_dim,
                                            nb_edge_types,
                                            use_3d,
                                            gnn_nb_layers,
                                            gnn_nb_hidden,
                                            mlp_nb_layers,
                                            mlp_nb_hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def forward(self):
        raise NotImplementedError

    def select_action(self, candidates, batch_idx):
        values = self.critic.get_value(candidates)
        shifted_actions = self.actor.select_action(
            torch.arange(len(batch_idx), device=batch_idx.device), values, batch_idx)
        actions = shifted_actions - get_batch_shift(batch_idx)

        return actions.squeeze_().tolist()

    def select_value(self, candidates, batch_idx):
        values = self.critic_target.get_value(candidates)
        qs = self.actor.select_action(values, values, batch_idx, eps=0)

        return qs, values

    def update(self, states, candidates, rewards, discounts, old_qs_next, old_values, batch_idx):
        if self.double_q:
            values = self.critic.get_value(candidates)
            qs_next = self.actor.select_action(old_values, values, batch_idx, eps=0)
        else:
            qs_next = old_qs_next
        loss = self.critic.loss(states, rewards, qs_next, discounts)

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
                 val_nb_layers,
                 val_nb_hidden):
        super(GAQN_Critic, self).__init__()
        self.gamma = gamma

        if not isinstance(gnn_nb_hidden, list):
            gnn_nb_hidden = [gnn_nb_hidden] * gnn_nb_layers
        else:
            assert len(gnn_nb_hidden) == gnn_nb_layers
        if not isinstance(val_nb_hidden, list):
            val_nb_hidden = [val_nb_hidden] * val_nb_layers
        else:
            assert len(val_nb_hidden) == val_nb_layers
            assert val_nb_layers > 0

        # gnn encoder (w/ 1 mlp layer)
        self.gnn = sGAT(input_dim, nb_edge_types, gnn_nb_hidden, gnn_nb_layers, 
                        output_dim=val_nb_hidden[0], use_3d=use_3d)
        in_dim = val_nb_hidden[0]

        # mlp encoder
        layers = []
        for i in range(1, val_nb_layers):
            layers.append(nn.Linear(in_dim, val_nb_hidden[i]))
            in_dim = val_nb_hidden[i]

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(in_dim, 1)
        self.act = nn.ReLU()

        self.MseLoss = nn.MSELoss()

    def forward(self, candidates):
        X = self.act(self.gnn(candidates))
        for l in self.layers:
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)

    def get_value(self, candidates):
        with torch.autograd.no_grad():
            values = self(candidates)
        return values.detach()

    def loss(self, states, rewards, qs_next, discounts):
        qs = self(states)
        targets = rewards + discounts * qs_next
        loss = self.MseLoss(qs, targets)

        return loss


class GAQN_Actor(nn.Module):
    def __init__(self, eps_clip):
        super(GAQN_Actor, self).__init__()
        self.eps_clip = eps_clip

    def forward(self):
        raise NotImplementedError

    def select_action(self, candidates, values, batch_idx, eps=None):
        if eps is None:
            eps = self.eps_clip
        probs = batched_argmax(values, batch_idx, eps=eps)
        shifted_actions = batched_sample(probs, batch_idx)

        return candidates[shifted_actions]
