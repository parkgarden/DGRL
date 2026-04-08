import time
import os
from common_utils import *
from params import configs
from data_utils import pack_data_from_config, pack_data_from_config2
from model.PPO import PPO_initialize
from common_utils import setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_dyn_env import FJSPEnvForDynGraph4
from fjsp_dyn_env_BA import FJSPEnvForDynGraphBA
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--model_source', type=str, default='SD2', help='Suffix of the data that model trained on')
parser.add_argument('--data_source', type=str, default='10020', help='Suffix of test data')
parser.add_argument('--dyn', type=int, default='0', help='dynamic mode')
parser.add_argument('--graph_len', type=int, default='5', help='dynamic graph length (number of operations per job to activate in the graph)')
parser.add_argument('--flag_sample', type=int, default='0', help='flag to sample actions or not')
parser.add_argument('--test_data', nargs='+', default=[''], help='List of data for testing')
parser.add_argument('--sample_times', type=int, default=100, help='Sampling times for the sampling strategy')
parser.add_argument('--test_model', nargs='+', default=['10x5+mix'], help='List of model for testing')
parser.add_argument('--n_ins', type=int, default=10, help='number of instances to test, 0 for all data')
parser.add_argument('--action_threshold', type=int, default=0, help='action probability threshold for MA')
parser.add_argument('--ins_start', type=int, default=0, help='start index of test data')
parser.add_argument('--ins_end', type=int, default=10, help='end index of test data')
params = parser.parse_args()


device = torch.device(configs.device)

ppo = PPO_initialize()
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def test_greedy_strategy(data_set, model_path, seed):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """
    test_result_list = []

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    env = FJSPEnvForDynGraphBA(n_j=n_j, n_m=n_m)

    activate_op_per_job = 10000
    action_threshold = 0
    dyn_ver = 0

    for i in range(len(data_set[0])):

        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]], activate_op_per_job)
        t1 = time.time()
        while True:
            with torch.no_grad():
                pi_, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
                                   op_mask=state.op_mask_tensor,  # [1, N, N]
                                   candidate=state.candidate_tensor,  # [1, J]
                                   fea_m=state.fea_m_tensor,  # [1, M, 6]
                                   mch_mask=state.mch_mask_tensor,  # [1, M, M]
                                   comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [1, J, M]
                                   fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

            pi = pi_.view(-1)
            n_m = state.fea_m_tensor.size(1)
            n_j = state.candidate_tensor.size(1)
            pi_reshaped = pi.view(n_j, n_m)
            max_select = min(n_j, n_m)

            if action_threshold == 0:
                valid_mask = pi_reshaped > action_threshold
            else:
                valid_mask = pi_reshaped > (1 / action_threshold)

            if valid_mask.sum() == 0 or valid_mask.sum() == 1:
                action = greedy_select_action(pi_)
                state, reward, done = env.step(dyn_ver, actions=action.cpu().numpy())

            else:
                # Multi-action selection via greedy bipartite matching
                valid = valid_mask.clone().to(device)
                probs = pi_reshaped
                NEG_INF = float('-inf')
                selected = []

                while True:
                    if not valid.any():
                        break
                    masked_probs = probs.clone()
                    masked_probs[~valid] = NEG_INF
                    flat_idx = int(masked_probs.view(-1).argmax().item())
                    if masked_probs.view(-1)[flat_idx].item() == NEG_INF:
                        break
                    j = flat_idx // n_m
                    m = flat_idx % n_m
                    selected.append(j * n_m + m)
                    valid[j, :] = False
                    valid[:, m] = False
                    if len(selected) >= max_select:
                        break

                best_actions = torch.tensor(selected)
                if best_actions.size(0) == 1:
                    state, reward, done = env.step(dyn_ver, actions=best_actions[0].unsqueeze(0).numpy())
                else:
                    state, reward, done = env.multi_step(dyn_ver, actions=best_actions.unsqueeze(0).numpy())

            if done:
                break
        t2 = time.time()

        test_result_list.append([env.current_makespan[0], t2 - t1])

    return np.array(test_result_list)



def test_dynBA_greedy_strategy(data_set, model_path, seed, dyn_ver, activate_op_per_job=10, action_threshold=0):
    """
        test the model on the given data using the greedy strategy with dynamic graph pruning
        and multi-action (BA) selection on benchmark instances
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :param dyn_ver: dynamic graph version
    :param activate_op_per_job: graph window size (operations per job visible in pruned graph)
    :param action_threshold: MA threshold denominator; 0 disables MA
    :return: the test results including the makespan and time
    """
    test_result_list = []

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    env = FJSPEnvForDynGraphBA(n_j=n_j, n_m=n_m)

    for i in range(len(data_set[0])):

        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]], activate_op_per_job)
        t1 = time.time()
        while True:
            with torch.no_grad():
                pi_, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
                                   op_mask=state.op_mask_tensor,  # [1, N, N]
                                   candidate=state.candidate_tensor,  # [1, J]
                                   fea_m=state.fea_m_tensor,  # [1, M, 6]
                                   mch_mask=state.mch_mask_tensor,  # [1, M, M]
                                   comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [1, J, M]
                                   fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

            pi = pi_.view(-1)
            n_m = state.fea_m_tensor.size(1)
            n_j = state.candidate_tensor.size(1)
            pi_reshaped = pi.view(n_j, n_m)
            max_select = min(n_j, n_m)

            if action_threshold == 0:
                valid_mask = pi_reshaped > action_threshold
            else:
                valid_mask = pi_reshaped > (1 / action_threshold)

            if valid_mask.sum() == 0 or valid_mask.sum() == 1:
                action = greedy_select_action(pi_)
                state, reward, done = env.step(dyn_ver, actions=action.cpu().numpy())

            else:
                # Multi-action selection via greedy bipartite matching
                valid = valid_mask.clone().to(device)
                probs = pi_reshaped
                NEG_INF = float('-inf')
                selected = []

                while True:
                    if not valid.any():
                        break
                    masked_probs = probs.clone()
                    masked_probs[~valid] = NEG_INF
                    flat_idx = int(masked_probs.view(-1).argmax().item())
                    if masked_probs.view(-1)[flat_idx].item() == NEG_INF:
                        break
                    j = flat_idx // n_m
                    m = flat_idx % n_m
                    selected.append(j * n_m + m)
                    valid[j, :] = False
                    valid[:, m] = False
                    if len(selected) >= max_select:
                        break

                best_actions = torch.tensor(selected)
                if best_actions.size(0) == 1:
                    state, reward, done = env.step(dyn_ver, actions=best_actions[0].unsqueeze(0).numpy())
                else:
                    state, reward, done = env.multi_step(dyn_ver, actions=best_actions.unsqueeze(0).numpy())

            if done:
                break
        t2 = time.time()

        test_result_list.append([env.current_makespan[0], t2 - t1])

    return np.array(test_result_list)


def test_sampling_strategy(data_set, model_path, sample_times, seed):
    """
        test the model on the given data using the sampling strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    test_result_list = []
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    n_op, n_m = data_set[1][0].shape
    env = FJSPEnvForSameOpNums(n_j, n_m)

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
        OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        t1 = time.time()
        while True:

            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
                                   op_mask=state.op_mask_tensor,  # [100, N, N]
                                   candidate=state.candidate_tensor,  # [100, J]
                                   fea_m=state.fea_m_tensor,  # [100, M, 6]
                                   mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                   comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                   fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

            action_envs, _ = sample_action(pi)
            state, _, done = env.step(action_envs.cpu().numpy())
            if done.all():
                break

        t2 = time.time()
        best_makespan = np.min(env.current_makespan)
        test_result_list.append([best_makespan, t2 - t1])

    return np.array(test_result_list)


def main(config, flag_sample):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    setup_seed(config.seed_test)
    os.makedirs('./test_results', exist_ok=True)

    # collect the path of test models
    test_model = []
    for model_name in config.test_model:
        test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))

    # collect the test data
    if params.test_data[0] == '0':
        test_data_input = ['']
    else:
        test_data_input = params.test_data
    test_data = pack_data_from_config2(params.data_source, test_data_input, params.ins_start, params.ins_end)

    if flag_sample:
        model_prefix = "DANIELS"
    else:
        model_prefix = "DANIELG"

    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{params.data_source}/{data[1]}'
        os.makedirs(save_direc, exist_ok=True)

        for model in test_model:
            if params.dyn == 0:
                save_path = save_direc + f'/Result_{model_prefix}+{model[1]}+DGRL_{data[1]}.npy'
            else:
                save_path = save_direc + f'/Result_{model_prefix}+{model[1]}+DGRL+len{params.graph_len}+dyn{params.dyn}+alpha{params.action_threshold}+ins{params.ins_start}-{params.ins_end}_{data[1]}.npy'
                print(save_path)

            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{params.data_source}/{data[1]}")

                # Map binary dyn flag to internal dyn_ver: 1 -> 15 (DGRL with invalid-job pruning)
                dyn_ver = 15 if params.dyn == 1 else params.dyn

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_5_times = []
                    for j in range(config.run):
                        if params.dyn == 0:
                            result = test_greedy_strategy(data[0], model[0], config.seed_test)
                        elif params.dyn >= 1:
                            result = test_dynBA_greedy_strategy(data[0], model[0], config.seed_test, dyn_ver, params.graph_len, params.action_threshold)
                        print(result)
                        result_5_times.append(result)
                    result_5_times = np.array(result_5_times)

                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print(f"makespan(greedy): ", save_result[:, 0].mean())
                    print(f"time: ", save_result[:, 1].mean())

                else:
                    print("Test mode: Sample")
                    save_result = test_sampling_strategy(data[0], model[0], config.sample_times, config.seed_test)
                    print("testing results:")
                    print(f"makespan(sampling): ", save_result[:, 0].mean())
                    print(f"time: ", save_result[:, 1].mean())
                np.save(save_path, save_result)


if __name__ == '__main__':
    main(configs, False)
