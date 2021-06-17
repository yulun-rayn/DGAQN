import logging
from tarfile import TarError

import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import Data, Batch

from .gaqn_value import TargetGAQN
from .rnd_explore import RNDistillation

#####################################################
#                   HELPER MODULES                  #
#####################################################

class Memory:
    def __init__(self):
        self.states = []        # state representations: pyg graph
        self.candidates = []    # next state (candidate) representations: pyg graph
        self.states_next = []   # next state (chosen) representations: pyg graph
        self.rewards = []       # rewards: float
        self.terminals = []     # trajectory status: logical

    def extend(self, memory):
        self.states.extend(memory.states)
        self.candidates.extend(memory.candidates)
        self.states_next.extend(memory.states_next)
        self.rewards.extend(memory.rewards)
        self.terminals.extend(memory.terminals)

    def clear(self):
        del self.states[:]
        del self.candidates[:]
        del self.states_next[:]
        del self.rewards[:]
        del self.terminals[:]

#################################################
#                  MAIN MODEL                   #
#################################################

class DGAQN(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 gamma,
                 eps_clip,
                 double_q,
                 k_epochs,
                 emb_model,
                 emb_nb_shared,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden,
                 enc_nb_output,
                 rnd_nb_layers,
                 rnd_nb_hidden,
                 rnd_nb_output):
        super(DGAQN, self).__init__()
        self.k_epochs = k_epochs
        self.use_3d = use_3d
        self.emb_model = emb_model
        self.emb_3d = emb_model.use_3d if emb_model is not None else use_3d
        self.emb_nb_shared = emb_nb_shared

        self.criterion = TargetGAQN(lr[0],
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
                                    enc_nb_layers,
                                    enc_nb_hidden,
                                    enc_nb_output)

        self.explore_critic = RNDistillation(lr[1],
                                             betas,
                                             eps,
                                             input_dim,
                                             nb_edge_types,
                                             use_3d,
                                             gnn_nb_layers,
                                             gnn_nb_hidden,
                                             rnd_nb_layers,
                                             rnd_nb_hidden,
                                             rnd_nb_output)

        self.device = torch.device("cpu")

    def to_device(self, device):
        if self.emb_model is not None:
            self.emb_model.to_device(device, n_layers=self.emb_nb_shared)
        self.criterion.to(device)
        self.explore_critic.to(device)
        self.device = device

    def forward(self):
        raise NotImplementedError

    def select_action(self, states, candidates, batch_idx=None):
        if batch_idx is None:
            batch_idx = torch.zeros(len(candidates), dtype=torch.long)
        batch_idx = torch.LongTensor(batch_idx).to(self.device)

        with torch.autograd.no_grad():
            if self.emb_model is not None:
                states = self.emb_model.get_embedding(states, n_layers=self.emb_nb_shared, return_3d=self.use_3d, aggr=False)
                candidates = self.emb_model.get_embedding(candidates, n_layers=self.emb_nb_shared, return_3d=self.use_3d, aggr=False)
            states_next, actions = self.criterion.select_state(candidates, batch_idx)

        if not isinstance(states, list):
            states = [states]
            candidates = [candidates]
            states_next = [states_next]
        states = [states[i].to_data_list() for i in range(1+self.use_3d)]
        states = list(zip(*states))
        candidates = [candidates[i].to_data_list() for i in range(1+self.use_3d)]
        candidates = list(zip(*candidates))
        states_next = [states_next[i].to_data_list() for i in range(1+self.use_3d)]
        states_next = list(zip(*states_next))

        return states, candidates, states_next, actions

    def get_inno_reward(self, states_next):
        if self.emb_model is not None:
            with torch.autograd.no_grad():
                states_next = self.emb_model.get_embedding(states_next, aggr=False)
        scores = self.explore_critic.get_score(states_next)
        return scores.squeeze().tolist()

    def update(self, memory, eps=1e-5):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, terminal in zip(reversed(memory.rewards), reversed(memory.terminals)):
            if terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        # candidates batch
        batch_idx = []
        for i, cands in enumerate(memory.candidates):
            batch_idx.extend([i]*len(cands))
        batch_idx = torch.LongTensor(batch_idx).to(self.device)

        # convert list to tensor
        states = [Batch().from_data_list([state[i] for state in memory.states]).to(self.device) 
                    for i in range(1+self.use_3d)]
        states_next = [Batch().from_data_list([state_next[i] for state_next in memory.states_next]).to(self.device) 
                    for i in range(1+self.use_3d)]
        candidates = [Batch().from_data_list([item[i] for sublist in memory.candidates for item in sublist]).to(self.device)
                        for i in range(1+self.use_3d)]
        terminals = torch.tensor(memory.terminals).to(self.device)

        old_qs_next, old_values = self.criterion.select_value(candidates, batch_idx)

        # Optimize value for k epochs:
        logging.info("Optimizing...")

        for i in range(self.k_epochs):
            loss = self.criterion.update(states, rewards, terminals, old_qs_next, old_values, batch_idx)
            rnd_loss = self.explore_critic.update(states_next)
            if (i%10)==0:
                logging.info("  {:3d}: DQN Loss: {:7.3f}, RND Loss: {:7.3f}".format(i, loss, rnd_loss))

        # Copy new weights into target network:
        self.criterion.update_target()

    def __repr__(self):
        return "{}\n".format(repr(self.criterion))
