import os
import gym
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict

import time
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from .DGAQN import DGAQN, Memory

from reward.get_main_reward import get_main_reward

from utils.general_utils import initialize_logger, close_logger, deque_to_csv
from utils.graph_utils import mols_to_pyg_batch

#####################################################
#                      PROCESS                      #
#####################################################

tasks = mp.JoinableQueue()
results = mp.Queue()


class Worker(mp.Process):
    def __init__(self, env, task_queue, result_queue, max_timesteps):
        super(Worker, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.env = env

        self.max_timesteps = max_timesteps
        self.timestep_counter = 0

    def run(self):
        # input:
        ## None:                                kill
        ## (None, None, True):                  dummy task
        ## (index, state, done):                trajectory id, molecule smiles, trajectory status
        #
        # output:
        ## (None, None, None, True):            dummy task
        ## (index, state, candidates, done):    trajectory id, molecule smiles, candidate smiles, trajectory status
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task == None:
                # Poison pill means shutdown
                print('\n%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            index, state, done = next_task
            if index is None:
                self.result_queue.put((None, None, None, True))
                self.task_queue.task_done()
                continue
            # print('%s: Working' % proc_name)
            if done:
                self.timestep_counter = 0
                state, candidates, done = self.env.reset(return_type='smiles')
            else:
                self.timestep_counter += 1
                state, candidates, done = self.env.reset(state, return_type='smiles')
                if self.timestep_counter >= self.max_timesteps:
                    done = True

            self.result_queue.put((index, state, candidates, done))
            self.task_queue.task_done()
        return

'''
class Task(object):
    def __init__(self, index, action, done):
        self.index = index
        self.action = action
        self.done = done
    def __call__(self):
        return (self.action, self.done)
    def __str__(self):
        return '%d' % self.index

class Result(object):
    def __init__(self, index, state, candidates, done):
        self.index = index
        self.state = state
        self.candidates = candidates
        self.done = done
    def __call__(self):
        return (self.state, self.candidates, self.done)
    def __str__(self):
        return '%d' % self.index
'''

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_serial(args, embed_model, env):
    lr = (args.dqn_lr, args.rnd_lr)
    betas = (args.beta1, args.beta2)
    eps = args.eps
    print("lr:", lr, "beta:", betas, "eps:", eps) # parameters for Adam optimizer

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    model = DGAQN(lr,
                betas,
                eps,
                args.gamma,
                args.eps_clip,
                args.double_q,
                args.k_epochs,
                embed_model,
                args.emb_nb_shared,
                args.input_size,
                args.nb_edge_types,
                args.use_3d,
                args.gnn_nb_layers,
                args.gnn_nb_hidden,
                args.enc_num_layers,
                args.enc_num_hidden,
                args.enc_num_output,
                args.rnd_num_layers,
                args.rnd_num_hidden,
                args.rnd_num_output)
    if args.running_model_path != '':
        model = torch.load(args.running_model_path)
    model.to_device(device)
    logging.info(model)

    time_step = 0

    avg_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=1000)
    # training loop
    for i_episode in range(1, args.max_episodes+1):
        state, candidates, done = env.reset()

        for t in range(args.max_timesteps):
            time_step += 1
            # Running policy:
            state_emb, candidates_emb, states_next_emb, action = model.select_state(
                mols_to_pyg_batch(state, model.emb_3d, device=model.device),
                mols_to_pyg_batch(candidates, model.emb_3d, device=model.device))
            memory.states.append(state_emb[0])
            memory.candidates.append(candidates_emb)
            memory.states_next.append(states_next_emb[0])

            state, candidates, done = env.step(action)

            # done and reward may not be needed anymore
            reward = 0

            if (t==(args.max_timesteps-1)) or done:
                main_reward = get_main_reward(state, reward_type=args.reward_type, args=args)
                reward = main_reward
                running_main_reward += main_reward

            if (args.iota > 0 and 
                i_episode > args.innovation_reward_episode_delay and 
                i_episode < args.innovation_reward_episode_cutoff):
                inno_reward = model.get_inno_reward(mols_to_pyg_batch(state, model.emb_3d, device=model.device))
                reward += inno_reward

            # Saving rewards and terminals:
            memory.rewards.append(reward)
            memory.terminals.append(done)

            running_reward += reward
            if done:
                break

        # update if it's time
        if time_step >= args.update_timesteps:
            logging.info("\nupdating model @ episode %d..." % i_episode)
            time_step = 0
            model.update(memory)
            memory.clear()

        writer.add_scalar("EpMainRew", main_reward, i_episode-1)
        rewbuffer_env.append(main_reward) # reward
        molbuffer_env.append((Chem.MolToSmiles(state), main_reward))
        avg_length += (t+1)

        # write to Tensorboard
        writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            torch.save(model, os.path.join(save_dir, 'DGAPN_continuous_solved_{}.pth'.format('test')))
            break

        # save every save_interval episodes
        if (i_episode-1) % args.save_interval == 0:
            torch.save(model, os.path.join(save_dir, '{:05d}_dgapn.pth'.format(i_episode)))
            deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'))

        # save running model
        torch.save(model, os.path.join(save_dir, 'running_dgapn.pth'))

        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length/args.log_interval)
            running_reward = running_reward/args.log_interval
            running_main_reward = running_main_reward/args.log_interval
            
            logging.info('Episode {} \t Avg length: {} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, avg_length, running_reward, running_main_reward))
            running_reward = 0
            running_main_reward = 0
            avg_length = 0

    close_logger()
    writer.close()
