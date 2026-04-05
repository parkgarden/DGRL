import argparse

def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def get_args():
    parser = argparse.ArgumentParser(description='Arguments for RL_GNN_JSP')
    # args for normal setting
    parser.add_argument('--device', type=str, default='cuda')
    # args for env
    parser.add_argument('--instance_type', type=str, default='FJSP')
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--max_process_time', type=int, default=100, help='Maximum Process Time of an Operation')
    parser.add_argument('--delete_node', type=bool, default=True)
    
    parser.add_argument('--dyn', type=int, default='0', help='dynamic mode')
    parser.add_argument('--graph_len', type=int, default='10', help='dynamic graph length (number of operations per job to activate in the graph)')
    # args for RL
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--episode', type=int, default=300001)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=float, default=1000)
    # args for policy network
    parser.add_argument('--hidden_dim', type=int, default=256) #256
    # args for GNN
    parser.add_argument('--GNN_num_layers', type=int, default=3)
    # args for policy
    parser.add_argument('--policy_num_layers', type=int, default=2)
    
    # args for nameing
    parser.add_argument('--date', type=str, default='Dummy')
    parser.add_argument('--detail', type=str, default="no")
    # args for structure
    parser.add_argument('--rule', type=str, default='MWKR')

    # args for val/test
    parser.add_argument('--test_dir', type=str, default='./datasets/FJSP/Brandimarte_Data')
    parser.add_argument('--load_weight', type=str, default='./weight/RS_FJSP/best')
    parser.add_argument('--data_source', type=str, default='Brandimarte_Data', help='Suffix of test data')
    parser.add_argument('--n_ins', type=int, default=10, help='number of instances to test, 0 for all data')
    parser.add_argument('--mode', type=int, default='0', help='ablation study mode')
    parser.add_argument('--data_char', type=int, default='0', help='0 : SD1, 1: SD2')
    parser.add_argument('--alpha', type=int, default='0', help='action probability threshold')
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
    args = parser.parse_args()
    return args
