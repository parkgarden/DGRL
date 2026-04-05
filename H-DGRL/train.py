import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
from heuristic import *
from torch.utils.tensorboard import SummaryWriter
import json
import time

import argparse
parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
# parser.add_argument('--data_source', type=str, default='10020', help='Suffix of test data')
parser.add_argument('--dyn', type=int, default='0', help='dynamic mode')
parser.add_argument('--graph_len', type=int, default='10', help='dynamic graph length (number of operations per job to activate in the graph)')
parser.add_argument('--data_char', type=int, default='0', help='0 : SD1, 1: SD2')
params = parser.parse_args()

MAX = float(1e6)

def train():
    print("start Training")
    best_valid_makespan = MAX

    for episode in range(0, args.episode):
        if episode == args.episode-1:
            torch.save(policy.state_dict(), "./weight/dyn{}_glen{}/{}".format(args.dyn, args.graph_len, episode))

        action_probs = []
        avai_ops = env.reset()
        while avai_ops is None:
            avai_ops = env.reset()

        MWKR_ms = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)

        while True:
            MWKR_baseline = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)
            baseline = MWKR_baseline - env.get_makespan()

            data, op_unfinished = env.get_graph_data()
            action_idx, action_prob = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time)
            avai_ops, reward, done = env.step(avai_ops[action_idx])

            policy.rewards.append(-reward)
            policy.baselines.append(baseline)
            action_probs.append(action_prob)
            
            if done:
                optimizer.zero_grad()
                loss, policy_loss, entropy_loss = policy.calculate_loss(args.device)
                loss.backward()

                if episode % 10 == 0:
                    writer.add_scalar("action prob", np.mean(action_probs),episode)
                    writer.add_scalar("loss", loss, episode)
                    writer.add_scalar("policy_loss", policy_loss, episode)
                    writer.add_scalar("entropy_loss", entropy_loss, episode)
                
                optimizer.step()
                scheduler.step()

                policy.clear_memory()
                ms = env.get_makespan()
                improve = MWKR_ms - ms
                if episode % 20 == 0:
                    print("Date : {} \t\t Episode : {} \t\tJob : {} \t\tMachine : {} \t\tPolicy : {} \t\tImprove: {} \t\t MWKR : {}".format(
                        args.date, episode, env.jsp_instance.job_num, env.jsp_instance.machine_num, 
                        ms, improve, MWKR_ms))
                    
                    if episode % 1000 == 0:
                        vali_result = validation(episode)
                        if vali_result < best_valid_makespan:
                            torch.save(policy.state_dict(), "./weight/dyn{}_glen{}/best".format(args.dyn, args.graph_len))
                            best_valid_makespan = vali_result
                break


def validation(episode, valid_sets=None):
    if args.instance_type == 'FJSP':
        valid_dir = './datasets/FJSP/data_dev'
        valid_sets = ['1510']

    else:
        valid_dir = './datasets/JSP/JSP_validation'
        valid_sets = ['20x20_valid']

    for _set in valid_sets:
        total_ms = 0.
        for instance in sorted(os.listdir(os.path.join(valid_dir, _set))):
            file = os.path.join(os.path.join(valid_dir, _set), instance)

            st = time.time()
            avai_ops = env.load_instance(file)

            while True:
                data, op_unfinished= env.get_graph_data()
                action_idx, _ = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time, greedy=True)
                avai_ops, _, done = env.step(avai_ops[action_idx])

                if done:
                    ed = time.time()
                    ms = env.get_makespan()
                    total_ms += ms
                    policy.clear_memory()

                    print('instance : {}, ms : {}, time : {}'.format(file, ms, ed - st))
                    break
        avg_ms = total_ms / len(os.listdir(os.path.join(valid_dir, _set)))
        with open('./result/dyn{}_glen{}/valid_result_{}.txt'.format(args.dyn, args.graph_len, _set),"a") as outfile:
            outfile.write(' set : {}, episode : {}, avg_ms : {}\n'.format(_set, episode, avg_ms))
    
    return avg_ms

if __name__ == '__main__':
    args = get_args()
    print(args)

    os.makedirs('./result/dyn{}_glen{}/'.format(args.dyn, args.graph_len), exist_ok=True)
    os.makedirs('./weight/dyn{}_glen{}/'.format(args.dyn, args.graph_len), exist_ok=True)

    with open("./result/dyn{}_glen{}/args.json".format(args.dyn, args.graph_len),"a") as outfile:
        json.dump(vars(args), outfile, indent=8)

    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.99)
    writer = SummaryWriter(comment=args.date)

    train()
    