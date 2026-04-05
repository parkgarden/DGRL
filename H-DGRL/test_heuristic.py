import numpy as np
import copy
import os
import re
import random
import torch
from params import get_args, str2bool
from env.env import JSP_Env
from env.dyn2_env import DFJSP_Env

from model.REINFORCE import REINFORCE
from heuristic import *
from torch.utils.tensorboard import SummaryWriter
import json
import time
from drawer import draw_gantt_chart
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--data_source', type=str, default='10020', help='Suffix of test data')
parser.add_argument('--dyn', type=int, default='0', help='dynamic mode')
parser.add_argument('--graph_len', type=int, default='10', help='dynamic graph length (number of operations per job to activate in the graph)')
parser.add_argument('--data_char', type=int, default='0', help='0 : SD1, 1: SD2')
parser.add_argument('--n_ins', type=int, default=10, help='number of instances to test, 0 for all data')
parser.add_argument('--plot_gantt', type=str2bool, default=False, help='Whether plotting gantt chart')

parser.add_argument('--dyn_job_arrival', type=str2bool, default=False, help='Dynamic environment with random job arrival')
parser.add_argument('--JA_interval_type', type=str, default='c', help='Job Arrival time distribution type (c:constant,p:poisson)')
parser.add_argument('--JA_interval', type=int, default=50, help='Avg new Job inter-arrival time')
parser.add_argument('--JA_num', type=int, default=5, help='Number of newly arrived jobs')

parser.add_argument('--dyn_machine_breakdown', type=str2bool, default=False, help='Dynamic environment with machine breakdown')
parser.add_argument('--MB_MTTR_dist', type=str, default='c', help='Mean time to repair distribution type (c:constant,p:poisson)')
parser.add_argument('--MB_MTTR', type=int, default=50, help='Mean time to repair')
parser.add_argument('--MB_numM', type=int, default=2, help='Mean time to repair')
parser.add_argument('--MB_interval', type=int, default=50, help='Mean time to repair')

parser.add_argument('--dyn_oper_add', type=str2bool, default=False, help='Dynamic environment with order additiona')
parser.add_argument('--OA_numJ', type=int, default=2, help='number of jobs to add operations')
parser.add_argument('--OA_numO', type=int, default=2, help='number of additional operations')
parser.add_argument('--dynEnv_state', type=str, default='0-0-0', help='Dynamic environment')

parser.add_argument('--run', type=int, default=5, help='number of experiment run')
params = parser.parse_args()

MAX = float(1e6)
def test(save_direc):
    test_dir = './datasets/FJSP/' + args.data_source
    files = os.listdir(test_dir)
    files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
    files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
    # breakpoint()
    cmax=[]
    cpu=[]
    if args.n_ins == 0:
        num_ins = len(files)
    else:
        num_ins = args.n_ins
    for instance in files[:num_ins]:
        file = os.path.join(test_dir, instance)
        # breakpoint()
        avai_ops = env.load_instance(file)
        st = time.time()
        data, op_unfinished= env.get_graph_data()
        # breakpoint()
        # k = 0
        MWKR_ms = heuristic_makespan(env, avai_ops, args.rule)
        ed = time.time()

        print("instance : {}, ms : {}, time : {}".format(file, MWKR_ms, ed - st))

        if args.plot_gantt:
            save_path = f'plot/dfjsp+{dyn_env_state}+{args.data_source}+ins{instance}+{args.rule}+_highlight3.png'
            draw_gantt_chart(env, save_path)
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)
        with open(save_direc + "test_result.txt", "a") as outfile:
            outfile.write(f'instance : {file:60}, policy : {MWKR_ms:10}\t')
            outfile.write(f'time : {ed - st:10}\n')
        cmax.append(env.get_makespan())
        cpu.append(ed - st)
        # break
        # cmax_results.append(torch.mean(torch.tensor(cmax)))
        # cpu_results.append(torch.mean(torch.tensor(cpu)))
    return cmax, cpu


if __name__ == '__main__':
    args = get_args()
    # print(args)
    
    if args.dyn_job_arrival or args.dyn_machine_breakdown or args.dyn_oper_add:
        env = DFJSP_Env(args)
    else:
        env = JSP_Env(args)

    dyn_env_state = ""
    if args.dyn_job_arrival:
        job_arrival_state = f"{args.JA_interval_type}{args.JA_interval}intJA{args.JA_num}"
        dyn_env_state += job_arrival_state
    if args.dyn_machine_breakdown:
        # machine_breakdown_state = f"p{configs.MB_prob}MB{configs.MB_MTTR_dist}{configs.MB_MTTR}MTTR"
        machine_breakdown_state = f"{args.MB_numM}MB{args.MB_interval}MTBF{args.MB_MTTR_dist}{args.MB_MTTR}MTTR" #random machine failures occurs every interval
        dyn_env_state += machine_breakdown_state
    if args.dyn_oper_add:
        # order_add_state = f"p{configs.order_reprocessing_prob}Reprocessing"
        # order_add_state = f"{configs.OA_numJ}J{configs.OA_numO}OAafter{configs.order_add_time}s"
        order_add_state = f"{args.OA_numJ}J{args.OA_numO}OA" # random operations insertion at the end of job after ith operation (i:random) 
        dyn_env_state += order_add_state
    
    save_direc = f'./result/{args.data_source}+{dyn_env_state}_PDR+{args.rule}/'

    cmax_results, cpu_results = test(save_direc)
    # breakpoint()
    data = pd.DataFrame({'Cmax': torch.tensor(cmax_results).t().tolist(), 'CPU':torch.tensor(cpu_results).t().tolist()})
    data.loc['Avg'] = data.mean()
    with pd.ExcelWriter('{}/{}_{}_{}result.xlsx'.format(save_direc, args.data_source, args.rule, dyn_env_state), engine="openpyxl") as writer:
        # breakpoint()
        data.to_excel(writer)