import torch
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
import time
import os

import argparse
parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--dyn', type=int, default='0', help='dynamic mode')
parser.add_argument('--graph_len', type=int, default='10', help='dynamic graph length (number of operations per job to activate in the graph)')
params = parser.parse_args()

MAX = float(1e6)

def eval_(episode, valid_sets=None):
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
        with open('./result/dyn{}_glen{}/valid_result2_{}.txt'.format(args.dyn, args.graph_len, _set),"a") as outfile:
            outfile.write(' set : {}, episode : {}, avg_ms : {}\n'.format(_set, episode, total_ms / len(os.listdir(os.path.join(valid_dir, _set)))))
        

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)

    os.makedirs('./result/dyn{}_glen{}/'.format(args.dyn, args.graph_len), exist_ok=True)
    files = os.listdir('./weight/dyn{}_glen{}/'.format(args.dyn, args.graph_len))
    files.remove('best')
    files.sort(key=lambda s: int(s))
    for episode in files:
        # if episode == 'best':
        #     continue
        print(f'date : {args.date} episode : {episode}')
        policy.load_state_dict(torch.load('./weight/dyn{}_glen{}/{}'.format(args.dyn, args.graph_len, episode), map_location=args.device), False)
        with torch.no_grad():
            valid_makespan = eval_(episode)