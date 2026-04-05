import copy
import json
import os
import random
import time as time

import gym
import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model_DGRL as PPO_model_DM
import PPO_model
import re
from env.load_data import nums_detec

import argparse

parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--data_source', type=str, default='10020', help='Suffix of test data')
parser.add_argument('--n_ins', type=int, default=10, help='number of instances to test, 0 for all data')
parser.add_argument('--ins_start', type=int, default=0, help='start index of test data')
parser.add_argument('--ins_end', type=int, default=10, help='end index of test data')
parser.add_argument('--ma', type=int, default=1, help='multi action selection')
parser.add_argument('--graph_len', type=int, default='10', help='dynamic graph length (number of operations per job to activate in the graph)')
parser.add_argument('--num_average', type=int, default=1, help='number of runs')
parser.add_argument('--alpha', type=int, default=0, help='action probability threshold for MA')
params = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = params.n_ins
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    test_dataset = params.data_source
    data_path = "./data_test/{0}/".format(test_dataset)

    test_files = os.listdir(data_path)
    test_files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
    test_files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
    test_files = test_files[params.ins_start:params.ins_end]
    mod_files = os.listdir('./model/DGRL/')[:]

    if params.ma == 1:
        memories = PPO_model_DM.Memory()
        model = PPO_model_DM.PPO(model_paras, train_paras)
    else:
        memories = PPO_model.Memory()
        model = PPO_model.PPO(model_paras, train_paras)
    rules = test_paras["rules"]
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    if "DRL" in rules:
        for root, ds, fs in os.walk('./model/DGRL/'):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")

    result_list = []
    data_name_list = []

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/DGRL/' + mod_files[i_rules])
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(params.ins_end - params.ins_start):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment if GPU memory usage exceeds 70%
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                # DRL-G, each env contains one instance
                else:
                    env = gym.make('fjspBA-v3', case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(i_ins))

            # Schedule an instance/environment
            # DRL-S
            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memories, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan))
                times.append(time_re)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                for j in range(test_paras["num_average"]):
                    makespan, time_re = schedule(env, model, memories, params.ma, params.alpha)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
            print("finish env {0}".format(i_ins))
        print("rule_spend_time: ", time.time() - step_time_last)

        data = pd.DataFrame({'Cmax': torch.tensor(makespans).t().tolist(), 'CPU':torch.tensor(times).t().tolist()})
        print(data)
        result_list.append(data)
        data_name_list.append(str(rule)[10:-3])

        for env in envs:
            env.reset()

    return result_list, data_name_list

def schedule(env, model, memories, ma, alpha, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()

    while ~done:
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, alpha, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)

    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!")

    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    test_dataset = params.data_source
    save_path = './save/test_{0}_DGRL_ma{1}_alpha{3}_ins{4}-{5}_{2}'.format(
        test_dataset, params.ma, str_time, params.alpha, params.ins_start, params.ins_end)
    os.makedirs(save_path)
    result_summary = []
    rule_list = []

    result_list, data_name_list = main()

    with pd.ExcelWriter('{0}/DGRL_{1}_ma{2}_alpha{3}_ins{4}-{5}_result.xlsx'.format(
            save_path, test_dataset, params.ma, params.alpha, params.ins_start, params.ins_end),
            engine="openpyxl") as writer:
        result_list, data_name_list = main()
        for i in range(len(result_list)):
            result_list[i].to_excel(writer, sheet_name=data_name_list[i])

            result_summary.append(result_list[i].mean(axis=0))
            rule_list.append(data_name_list[i])
        mean_df = pd.concat(result_summary, axis=1)
        mean_df.columns = rule_list
        mean_df.to_excel(writer, sheet_name='Avg')
