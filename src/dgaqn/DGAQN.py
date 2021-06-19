import logging
from tarfile import TarError

import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import Data, Batch

from gnn_embed import init_sGAT

from .gaqn_value import TargetGAQN
from .rnd_explore import RNDistillation

#####################################################
#                   HELPER MODULES                  #
#####################################################

def init_DGAQN(state):
    net = DGAQN(state['lr'],
                state['betas'],
                state['eps'],
                state['gamma'],
                state['eps_clip'],
                state['double_q'],
                state['k_epochs'],
                state['emb_state'],
                state['emb_nb_shared'],
                state['input_dim'],
                state['nb_edge_types'],
                state['use_3d'],
                state['gnn_nb_layers'],
                state['gnn_nb_hidden'],
                state['val_nb_layers'],
                state['val_nb_hidden'],
                state['rnd_nb_layers'],
                state['rnd_nb_hidden'],
                state['rnd_nb_output'])
    net.load_state_dict(state['state_dict'])
    return net

def load_DGAQN(state_path):
    state = torch.load(state_path)
    return init_DGAQN(state)

def save_DGAQN(net, state_path=None):
    torch.save(net.get_dict(), state_path)

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
                 emb_state,
                 emb_nb_shared,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 val_nb_layers,
                 val_nb_hidden,
                 rnd_nb_layers,
                 rnd_nb_hidden,
                 rnd_nb_output):
        super(DGAQN, self).__init__()
        if emb_state is not None:
            emb_model = init_sGAT(emb_state)
            print("embed model loaded")
            emb_model.eval()
            print(emb_model)
        else:
            emb_model = None
        self.emb_model = emb_model
        self.emb_3d = emb_model.use_3d if emb_model is not None else use_3d

        self.lr=lr
        self.betas=betas
        self.eps=eps
        self.gamma=gamma
        self.eps_clip=eps_clip
        self.double_q=double_q
        self.k_epochs=k_epochs
        self.emb_state=emb_state
        self.emb_nb_shared=emb_nb_shared
        self.input_dim=input_dim
        self.nb_edge_types=nb_edge_types
        self.use_3d=use_3d
        self.gnn_nb_layers=gnn_nb_layers
        self.gnn_nb_hidden=gnn_nb_hidden
        self.val_nb_layers=val_nb_layers
        self.val_nb_hidden=val_nb_hidden
        self.rnd_nb_layers=rnd_nb_layers
        self.rnd_nb_hidden=rnd_nb_hidden
        self.rnd_nb_output=rnd_nb_output

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
                                    val_nb_layers,
                                    val_nb_hidden)

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
            actions = self.criterion.select_action(candidates, batch_idx)

        if not isinstance(states, list):
            states = [states]
            candidates = [candidates]
        states = [states[i].to_data_list() for i in range(1+self.use_3d)]
        states = list(zip(*states))
        candidates = [candidates[i].to_data_list() for i in range(1+self.use_3d)]
        candidates = list(zip(*candidates))

        return states, candidates, actions

    def get_inno_reward(self, states_next):
        if self.emb_model is not None:
            with torch.autograd.no_grad():
                states_next = self.emb_model.get_embedding(states_next, aggr=False)
        scores = self.explore_critic.get_score(states_next)
        return scores.squeeze().tolist()

    def update(self, memory, eps=1e-5):
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
        rewards = torch.tensor(memory.rewards).to(self.device)
        discounts = self.gamma * ~torch.tensor(memory.terminals).to(self.device)

        old_qs_next, old_values = self.criterion.select_value(candidates, batch_idx)

        # Optimize value for k epochs:
        logging.info("Optimizing...")

        for i in range(1, self.k_epochs+1):
            loss = self.criterion.update(states, candidates, rewards, discounts, old_qs_next, old_values, batch_idx)
            rnd_loss = self.explore_critic.update(states_next)
            if (i%10)==0:
                logging.info("  {:3d}: DQN Loss: {:7.3f}, RND Loss: {:7.3f}".format(i, loss, rnd_loss))

        # Copy new weights into target network:
        self.criterion.update_target()

    def get_dict(self):
        state = {'state_dict': self.state_dict(),
                    'lr': self.lr,
                    'betas': self.betas,
                    'eps': self.eps,
                    'gamma': self.gamma,
                    'eps_clip': self.eps_clip,
                    'double_q': self.double_q,
                    'k_epochs': self.k_epochs,
                    'emb_state': self.emb_state,
                    'emb_nb_shared': self.emb_nb_shared,
                    'input_dim': self.input_dim,
                    'nb_edge_types': self.nb_edge_types,
                    'use_3d': self.use_3d,
                    'gnn_nb_layers': self.gnn_nb_layers,
                    'gnn_nb_hidden': self.gnn_nb_hidden,
                    'val_nb_layers': self.val_nb_layers,
                    'val_nb_hidden': self.val_nb_hidden,
                    'rnd_nb_layers': self.rnd_nb_layers,
                    'rnd_nb_hidden': self.rnd_nb_hidden,
                    'rnd_nb_output': self.rnd_nb_output}
        return state

    def __repr__(self):
        return "{}\n".format(repr(self.criterion))
