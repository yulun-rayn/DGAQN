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
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr[1], betas=betas, eps=eps)

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
        shifted_actions = self.actor.select_action(range(len(batch_idx)), values, batch_idx)
        states_next = candidates[shifted_actions]
        actions = shifted_actions - get_batch_shift(batch_idx)

        return states_next, actions.squeeze_().tolist()

    def select_value(self, candidates, batch_idx):
        values = self.critic_target.get_value(candidates)
        qs_next = self.actor.select_action(values, values, batch_idx)

        return qs_next, values

    def get_value(self, candidates):
        return self.critic_target.get_value(candidates)

    def update(self, states, candidates, rewards, terminals, old_qs_next, old_values, batch_idx):
        if self.double_q:
            values = self.critic.get_value(candidates)
            qs_next = self.actor.select_action(old_values, values, batch_idx)
        else:
            qs_next = old_qs_next
        loss = self.critic.loss(states, rewards, qs_next, terminals)

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

        self.gnn = sGAT(input_dim, gnn_nb_hidden, gnn_nb_layers, nb_edge_types, use_3d=use_3d)
        if gnn_nb_layers == 0:
            in_dim = input_dim
        else:
            in_dim = gnn_nb_hidden

        layers = []
        for _ in range(enc_nb_layers):
            layers.append(nn.Linear(in_dim, enc_nb_hidden))
            in_dim = enc_nb_hidden

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(in_dim, 1)
        self.act = nn.ReLU()

        self.MseLoss = nn.MSELoss()

    def forward(self, candidates):
        X = self.gnn.get_embedding(candidates, detach=False)
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)

    def get_value(self, candidates):
        with torch.autograd.no_grad():
            values = self(candidates)
        return values.detach()

    def loss(self, states, rewards, qs_next, terminals):
        qs = self(states)
        targets = rewards + self.gamma * qs_next * (1 - terminals)
        loss = self.MseLoss(qs, targets)

        return loss


class GAQN_Actor(nn.Module):
    def __init__(self, eps_clip):
        super(GAQN_Actor, self).__init__()
        self.eps_clip = eps_clip

    def forward(self):
        raise NotImplementedError

    def select_action(self, candidates, values, batch_idx):
        probs = batched_argmax(values, batch_idx, eps=self.eps_clip)
        shifted_actions = batched_sample(probs, batch_idx)

        return candidates[shifted_actions]
