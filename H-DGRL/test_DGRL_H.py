import torch
from params import get_args
from env.dyn_env_BA2 import JSP_Env
from model.REINFORCE import REINFORCE
from model.REINFORCE_DGRL import REINFORCE_DM
import time
import os
import re

import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--data_source', type=str, default='Brandimarte_Data', help='Suffix of test data')
parser.add_argument('--dyn', type=int, default='0', help='dynamic mode')
parser.add_argument('--graph_len', type=int, default='10', help='dynamic graph length (number of operations per job to activate in the graph)')
parser.add_argument('--n_ins', type=int, default=10, help='number of instances to test, 0 for all data')
parser.add_argument('--mode', type=int, default='0', help='mode: 0=single-action, 1=multi-action with DG, 2=multi-action w/o DG, 3=single-action w/o MA')
parser.add_argument('--alpha', type=int, default='0', help='action probability threshold')
params = parser.parse_args()

def test(runs):
    test_dir = './datasets/FJSP/' + args.data_source
    files = os.listdir(test_dir)
    files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
    files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
    cmax_results = []
    cpu_results = []
    num_ins = len(files) if args.n_ins == 0 else args.n_ins
    for instance in files[:num_ins]:
        file = os.path.join(test_dir, instance)
        cmax = []
        cpu = []
        for i in range(runs):
            avai_ops = env.load_instance(file)
            st = time.time()
            data, op_unfinished = env.get_graph_data()
            while True:
                if args.mode == 0 or args.mode == 3:
                    # Single-action mode
                    action_idx, action_prob = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time, greedy=True)
                    avai_ops, _, done = env.step(avai_ops[action_idx])
                    if done:
                        ed = time.time()
                        policy.clear_memory()
                        print("instance : {}, ms : {}, time : {}".format(file, env.get_makespan(), ed - st))
                        save_direc = f'./result/{args.data_source}_dyn{args.dyn}_glen{args.graph_len}/'
                        os.makedirs(save_direc, exist_ok=True)
                        with open(save_direc + "test_DGRL_single_result.txt", "a") as outfile:
                            outfile.write(f'instance : {file:60}, policy : {env.get_makespan():10}\t')
                            outfile.write(f'time : {ed - st:10}\n')
                        cmax.append(env.get_makespan())
                        cpu.append(ed - st)
                        break
                    else:
                        data, op_unfinished = env.get_graph_data()
                else:
                    # Multi-action mode
                    actions, action_prob = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time, greedy=True, mode=args.mode, alpha=args.alpha)
                    avai_ops, _, done = env.step(actions)
                    if done:
                        ed = time.time()
                        policy.clear_memory()
                        print("instance : {}, ms : {}, time : {}".format(file, env.get_makespan(), ed - st))
                        save_direc = f'./result/{args.data_source}_dyn{args.dyn}_glen{args.graph_len}/'
                        os.makedirs(save_direc, exist_ok=True)
                        with open(save_direc + "test_DGRL_result.txt", "a") as outfile:
                            outfile.write(f'instance : {file:60}, policy : {env.get_makespan():10}\t')
                            outfile.write(f'time : {ed - st:10}\n')
                        cmax.append(env.get_makespan())
                        cpu.append(ed - st)
                        break
                    else:
                        data, op_unfinished = env.get_graph_data()
        cmax_results.append(torch.mean(torch.tensor(cmax)))
        cpu_results.append(torch.mean(torch.tensor(cpu)))

    return cmax_results, cpu_results

if __name__ == '__main__':
    args = get_args()
    env = JSP_Env(args)
    mode = args.mode  # 0: single-action, 1: MA with DG, 2: MA w/o DG, 3: single-action w/o MA
    if mode == 0 or mode == 2:
        policy = REINFORCE(args).to(args.device)
    else:
        policy = REINFORCE_DM(args).to(args.device)

    trained_network = './weight/dyn{}_glen{}/'.format(args.dyn, args.graph_len) + 'best'
    save_direc = f'./result/{args.data_source}/'
    os.makedirs(save_direc, exist_ok=True)
    policy.load_state_dict(torch.load(trained_network, map_location=args.device), False)
    with torch.no_grad():
        cmax_results, cpu_results = test(5)

    data = pd.DataFrame({'Cmax': torch.tensor(cmax_results).t().tolist(), 'CPU': torch.tensor(cpu_results).t().tolist()})
    data.loc['Avg'] = data.mean()
    if args.dyn == 0:
        with pd.ExcelWriter('{}/{}_DGRL_result.xlsx'.format(save_direc, args.data_source), engine="openpyxl") as writer:
            data.to_excel(writer)
    else:
        with pd.ExcelWriter('{}/{}_dyn{}_glen{}_mode{}_alpha{}_DGRL_result.xlsx'.format(
                save_direc, args.data_source, args.dyn, args.graph_len, args.mode, args.alpha),
                engine="openpyxl") as writer:
            data.to_excel(writer)
