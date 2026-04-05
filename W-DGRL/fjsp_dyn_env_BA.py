from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import copy
from params import configs
import sys
import torch


@dataclass
class EnvState:
    """
        state definition
    """
    fea_j_tensor: torch.Tensor = None
    op_mask_tensor: torch.Tensor = None
    fea_m_tensor: torch.Tensor = None
    mch_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None
    comp_idx_tensor: torch.Tensor = None
    candidate_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None

    device = torch.device(configs.device)

    def update(self, fea_j, op_mask, fea_m, mch_mask, dynamic_pair_mask,
               comp_idx, candidate, fea_pairs):
        """
            update the state information
        :param fea_j: input operation feature vectors with shape [sz_b, N, 10]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 8]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
        """
        device = self.device
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device)
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.op_mask_tensor = torch.from_numpy(np.copy(op_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.mch_mask_tensor = torch.from_numpy(np.copy(mch_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)

    def print_shape(self):
        print(self.fea_j_tensor.shape)
        print(self.op_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_m_tensor.shape)
        print(self.mch_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)


class FJSPEnvForDynGraphBA:
    """
        a batch of fjsp environments that have the same number of operations

        let E/N/J/M denote the number of envs/operations/jobs/machines
        Remark: The index of operations has been rearranged in natural order
        eg. {O_{11}, O_{12}, O_{13}, O_{21}, O_{22}}  <--> {0,1,2,3,4}

        Attributes:

        job_length: the number of operations in each job (shape [J])
        op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$

        candidate: the index of candidates  [sz_b, J]
        fea_j: input operation feature vectors with shape [sz_b, N, 8]
        op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        fea_m: input operation feature vectors with shape [sz_b, M, 6]
        mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking incompatible op-mch pairs
        fea_pairs: pair features with shape [sz_b, J, M, 8]
    """

    def __init__(self, n_j, n_m):
        """
        :param n_j: the number of jobs
        :param n_m: the number of machines
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()

        # the dimension of operation raw features
        self.op_fea_dim = 10
        # the dimension of machine raw features
        self.mch_fea_dim = 8

    def set_static_properties(self):
        """
            define static properties
        """
        self.multi_env_mch_diag = np.tile(np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)
        self.op_idx = np.arange(self.number_of_ops)[np.newaxis, :]

    def set_initial_data(self, job_length_list, op_pt_list, activate_num_op):
        """
            initialize the data of the instances

        :param job_length_list: the list of 'job_length'
        :param op_pt_list: the list of 'op_pt'
        """
        self.activate_num_op = activate_num_op

        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.op_pt = np.array(op_pt_list)
        self.number_of_ops = self.op_pt.shape[1]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        self.set_static_properties()

        # [E, N, M]
        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)

        # normalize the processing time
        self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # bool 3-d array formulating the compatible relation with shape [E,N,M]
        self.process_relation = (self.op_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        # number of compatible machines of each operation ([E,N])
        self.compatible_op = np.sum(self.process_relation, 2)
        # number of operations that each machine can process ([E,M])
        self.compatible_mch = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)

        head_op_id = np.zeros((self.number_of_envs, 1))

        # the index of first operation of each job ([E,J])
        self.job_first_op_id = np.concatenate([head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype(
            'int')
        # the index of last operation of each job ([E,J])
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        self.initial_vars()

        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)

        """
            compute operation raw features
        """
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data

        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt
        # [E, M]
        self.mch_min_pt = np.min(self.op_pt, axis=1).data
        self.mch_max_pt = np.max(self.op_pt, axis=1).data

        # the estimated lower bound of complete time of operations
        self.op_ct_lb = copy.deepcopy(self.op_min_pt)
        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1] = np.cumsum(
                    self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])

        # job remaining number of operations
        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.job_length[k])
                                                   for k in range(self.number_of_envs)])
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])

        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.job_length[k])
                                                  for k in range(self.number_of_envs)])

        self.construct_op_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        """
            compute machine raw features
        """
        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        # [E, J, M]
        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        # construct dynamic pair mask : [E, J, M]
        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        # construct machine features [E, M, 6]

        # construct 'come_idx' : [E, M, M, J]
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.init_mch_mask()
        self.construct_mch_features(self.delete_mask_fea_m)

        self.construct_pair_features()

        self.old_state.update(self.fea_j, self.op_mask,
                              self.fea_m, self.mch_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs)

        # old record
        self.old_op_mask = np.copy(self.op_mask)
        self.old_mch_mask = np.copy(self.mch_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        """
           reset the environments
        :return: the state
        """
        self.initial_vars()

        # copy the old data
        self.op_mask = np.copy(self.old_op_mask)
        self.mch_mask = np.copy(self.old_mch_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(self.old_mch_current_available_op_nums)
        self.mch_current_available_jc_nums = np.copy(self.old_mch_current_available_jc_nums)
        # copy the old state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        """
            initialize variables for further use
        """
        self.step_count = 0
        # the array that records the makespan of all environments
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        # the complete time of operations ([E,N])
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        # mask[i,j] : whether the jth job of ith env is scheduled (have no unscheduled operations)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.number_of_ops))

        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs,
                                        self.number_of_machines))
        self.true_pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs,
                                        self.number_of_machines))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.number_of_ops, self.op_fea_dim),
                                         fill_value=0, dtype=bool)
        # mask[i,j] : whether the jth op of ith env is deleted (from the set $O_u$)
        self.deleted_op_nodes = np.full(shape=(self.number_of_envs, self.number_of_ops),
                                        fill_value=0, dtype=bool)
        
        self.deleted_future_op_nodes = np.full(shape=(self.number_of_envs, self.number_of_ops),
                                        fill_value=0, dtype=bool)
        
        # Initialize self.deleted_future_op_nodes to 1 (all operations initially deleted)
        self.deleted_future_op_nodes = np.ones((self.number_of_envs, self.number_of_ops), dtype=bool)

        # Loop through each environment and set the specified range to 0
        for env_idx in range(self.number_of_envs):
            for job_idx in range(self.job_first_op_id.shape[1]):
                start_idx = self.job_first_op_id[env_idx, job_idx]  # Start of the job
                end_idx = min(start_idx + self.activate_num_op, self.job_last_op_id[env_idx, job_idx])  # End index for activation
                self.deleted_future_op_nodes[env_idx, start_idx:end_idx] = 0  # Set active operations to 0


        job_start_indices = self.job_first_op_id[self.env_idxs, :]
        job_end_indices = self.job_last_op_id[self.env_idxs, :] + 1  # Inclusive of end index

        op_ranges = torch.arange(self.number_of_ops).unsqueeze(0).expand(self.number_of_envs, -1)
        start_expanded = torch.tensor(job_start_indices).unsqueeze(-1)  # Shape: [num_environments, num_jobs, 1]
        end_expanded = torch.tensor(job_end_indices).unsqueeze(-1)  # Shape: [num_environments, num_jobs, 1]

        # Mask operations based on job range & invalid jobs
        self.job_op_match_mask = (op_ranges.unsqueeze(1) >= start_expanded) & (op_ranges.unsqueeze(1) < end_expanded)

        self.total_deleted_op_nodes = np.logical_or(self.deleted_op_nodes, self.deleted_future_op_nodes)


    def step(self,dyn_ver, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]

        # if chosen_op == 1149 and chosen_job == 56 and chosen_mch == 9:

        if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
        self.mask[self.env_idxs, chosen_job] = (1 - candidate_add_flag)

        # the start processing time of chosen operations
        chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job],
                                  self.mch_free_time[self.env_idxs, chosen_mch])

        self.op_ct[self.env_idxs, chosen_op] = chosen_op_st + self.op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
        self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],
                                       self.true_mch_free_time[self.env_idxs, chosen_mch])
        self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op]

        self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[
            self.env_idxs, chosen_op])

        # update the candidate message
        mask_temp = candidate_add_flag
        self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = \
            self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_process_relation[~mask_temp, chosen_job[~mask_temp]] = 1

        # compute the next schedule time

        # [E, J, M]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        self.remain_process_relation[self.env_idxs, chosen_op] = 0
        self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

        """
            update the mask for deleting nodes
        """
        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)
        # self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
        #                                 (1, 1, self.op_fea_dim))

        if dyn_ver != 0:
            activate_future_op_node = np.minimum(chosen_op + self.activate_num_op, self.job_last_op_id[self.env_idxs, chosen_job])

            # self.deleted_future_op_nodes[self.env_idxs, activate_future_op_node] = 0 
            # self.total_deleted_op_nodes = np.logical_or(self.deleted_op_nodes, self.deleted_future_op_nodes) 
            # self.delete_mask_fea_j = np.tile(self.total_deleted_op_nodes[:, :, np.newaxis],
            #                                 (1, 1, self.op_fea_dim))
            
            if dyn_ver != 14:
                invalid_job_mask = torch.tensor(self.candidate_process_relation).all(dim=2) #shape torch.Size([num_enviroments, num_jobs])
                invalid_op_mask = torch.zeros((self.number_of_envs, self.number_of_ops), dtype=torch.bool)
                
                invalid_op_mask |= torch.any(self.job_op_match_mask * invalid_job_mask.unsqueeze(-1), dim=1)
            
            np.put_along_axis(
                self.deleted_future_op_nodes,
                activate_future_op_node[:, np.newaxis],
                0,
                axis=1
            )

            # Combine deleted nodes
            self.total_deleted_op_nodes = np.logical_or(
                self.deleted_op_nodes,
                self.deleted_future_op_nodes,
            )
            if dyn_ver != 14:
                self.total_deleted_op_nodes = np.logical_or(
                    self.total_deleted_op_nodes,
                    invalid_op_mask.cpu().numpy(),
                )

            # Efficient broadcasting for delete_mask_fea_j
            self.delete_mask_fea_j = np.broadcast_to(
                self.total_deleted_op_nodes[:, :, np.newaxis],
                (self.total_deleted_op_nodes.shape[0], self.total_deleted_op_nodes.shape[1], self.op_fea_dim)
            )
        else:
            self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))

        """
            update the state
        """
        pruned_op_mask = self.update_op_mask(dyn_ver)

        # update operation raw features
        diff = self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]

        mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                               np.newaxis])
        self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]

        mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:,
                                np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                               np.newaxis])
        self.op_match_job_left_op_nums[mask2] -= 1
        self.op_match_job_remain_work[mask2] -= \
            np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[mask2]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)
        
        self.construct_op_features()

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # update comp_idx
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_mch_mask()

        # update machine raw features
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums -= self.process_relation[
            self.env_idxs, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        mch_delete_mask =  np.copy(self.delete_mask_fea_m)
        if dyn_ver == 14 or dyn_ver == 15:
            invalid_mch_mask = torch.tensor(self.candidate_process_relation).all(dim=1) #shape torch.Size([num_enviroments, num_machine])
            mch_delete_mask[mch_free_flag] = 1

        self.construct_mch_features(mch_delete_mask)

        self.construct_pair_features()

        if dyn_ver == 10 or dyn_ver == 12:
            fea_pairs = self.fea_pairs.copy()
            fea_pairs[invalid_job_mask.cpu(), :] = 0
        else:
            fea_pairs = self.fea_pairs
            

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        # update the state
        self.state.update(self.fea_j, pruned_op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          fea_pairs)

        return self.state, np.array(reward), self.done()
    
    # def multi_step(self,dyn_ver, actions):
    #     # breakpoint()
    #     for a in range(actions.size):
    #         state, reward, done = self.step(dyn_ver, np.array([actions[0,a]]))
    #     return state, reward, done

    def multi_step2(self,dyn_ver, actions):
        for a in range(actions.size):
            action =  np.array([actions[0,a]])
            chosen_job = action // self.number_of_machines
            chosen_mch = action % self.number_of_machines
            chosen_op = self.candidate[self.env_idxs, chosen_job]

            # if chosen_op == 1149 and chosen_job == 56 and chosen_mch == 9:

            if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
                print(
                    f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
                sys.exit()

            self.step_count += 1

            # update candidate
            candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
            self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
            self.mask[self.env_idxs, chosen_job] = (1 - candidate_add_flag)

            # the start processing time of chosen operations
            chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job],
                                    self.mch_free_time[self.env_idxs, chosen_mch])

            self.op_ct[self.env_idxs, chosen_op] = chosen_op_st + self.op_pt[
                self.env_idxs, chosen_op, chosen_mch]
            self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
            self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]

            true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],
                                        self.true_mch_free_time[self.env_idxs, chosen_mch])
            self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
                self.env_idxs, chosen_op, chosen_mch]
            self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
                self.env_idxs, chosen_op]
            self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
                self.env_idxs, chosen_op]

            self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[
                self.env_idxs, chosen_op])

            # update the candidate message
            mask_temp = candidate_add_flag
            self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[mask_temp, chosen_op[mask_temp] + 1]
            self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = \
                self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1]
            self.candidate_process_relation[~mask_temp, chosen_job[~mask_temp]] = 1

            # compute the next schedule time

            # [E, J, M]
            candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
            mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
            self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

            schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)

            self.next_schedule_time = np.min(
                schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

            self.remain_process_relation[self.env_idxs, chosen_op] = 0
            self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

            """
                update the mask for deleting nodes
            """
            self.deleted_op_nodes = \
                np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                            self.op_scheduled_flag)
            
            if dyn_ver == 0:
                self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                                (1, 1, self.op_fea_dim))
            
            else:
                activate_future_op_node = np.minimum(chosen_op + self.activate_num_op, self.job_last_op_id[self.env_idxs, chosen_job])

                # self.deleted_future_op_nodes[self.env_idxs, activate_future_op_node] = 0 
                # self.total_deleted_op_nodes = np.logical_or(self.deleted_op_nodes, self.deleted_future_op_nodes) 
                # self.delete_mask_fea_j = np.tile(self.total_deleted_op_nodes[:, :, np.newaxis],
                #                                 (1, 1, self.op_fea_dim))
                
                if dyn_ver != 14:
                    invalid_job_mask = torch.tensor(self.candidate_process_relation).all(dim=2) #shape torch.Size([num_enviroments, num_jobs])
                    invalid_op_mask = torch.zeros((self.number_of_envs, self.number_of_ops), dtype=torch.bool)
                    
                    invalid_op_mask |= torch.any(self.job_op_match_mask * invalid_job_mask.unsqueeze(-1), dim=1)
                
                np.put_along_axis(
                    self.deleted_future_op_nodes,
                    activate_future_op_node[:, np.newaxis],
                    0,
                    axis=1
                )

                # Combine deleted nodes
                self.total_deleted_op_nodes = np.logical_or(
                    self.deleted_op_nodes,
                    self.deleted_future_op_nodes,
                )
                if dyn_ver != 14:
                    self.total_deleted_op_nodes = np.logical_or(
                        self.total_deleted_op_nodes,
                        invalid_op_mask.cpu().numpy(),
                    )

                # Efficient broadcasting for delete_mask_fea_j
                self.delete_mask_fea_j = np.broadcast_to(
                    self.total_deleted_op_nodes[:, :, np.newaxis],
                    (self.total_deleted_op_nodes.shape[0], self.total_deleted_op_nodes.shape[1], self.op_fea_dim)
                )
            

            """
                update the state
            """
            pruned_op_mask = self.update_op_mask(dyn_ver)

            # update operation raw features
            diff = self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]

            mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & \
                    (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                                np.newaxis])
            self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]

            mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:,
                                    np.newaxis]) & \
                    (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                                np.newaxis])
            self.op_match_job_left_op_nums[mask2] -= 1
            self.op_match_job_remain_work[mask2] -= \
                np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[mask2]

            self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
            self.op_waiting_time[self.env_job_idx, self.candidate] = \
                (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                            - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                    self.env_job_idx, self.candidate]

            self.op_remain_work = np.maximum(self.op_ct -
                                            np.expand_dims(self.next_schedule_time, axis=1), 0)
            
            self.construct_op_features()

            # update dynamic pair mask
            self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

            self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

            self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

            # update comp_idx
            self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

            self.update_mch_mask()

            # update machine raw features
            self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
            self.mch_current_available_op_nums -= self.process_relation[
                self.env_idxs, chosen_op]

            mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
            mch_free_flag = mch_free_duration < 0
            self.mch_working_flag = mch_free_flag + 0
            self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

            self.mch_remain_work = np.maximum(-mch_free_duration, 0)

            mch_delete_mask =  np.copy(self.delete_mask_fea_m)
            if dyn_ver == 14 or dyn_ver == 15:
                invalid_mch_mask = torch.tensor(self.candidate_process_relation).all(dim=1) #shape torch.Size([num_enviroments, num_machine])
                mch_delete_mask[mch_free_flag] = 1

            self.construct_mch_features(mch_delete_mask)

            self.construct_pair_features()

            if dyn_ver == 10 or dyn_ver == 12:
                fea_pairs = self.fea_pairs.copy()
                fea_pairs[invalid_job_mask.cpu(), :] = 0
            else:
                fea_pairs = self.fea_pairs
                

            # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
            reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
            self.max_endTime = np.max(self.op_ct_lb, axis=1)

            # update the state
            self.state.update(self.fea_j, pruned_op_mask, self.fea_m, self.mch_mask,
                            self.dynamic_pair_mask, self.comp_idx, self.candidate,
                            fea_pairs)

        # return self.state, np.array(reward), self.done()
        return self.state, np.array(reward), self.done()

    
    def multi_step(self, dyn_ver, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_job2 = actions // self.number_of_machines
        chosen_mch2 = actions % self.number_of_machines
        chosen_op2 = self.candidate[self.env_idxs, chosen_job2]

        # if chosen_job == 18:

        # if chosen_op == 1149 and chosen_job == 56 and chosen_mch == 9:

        # if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
        #     print(
        #         f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
        #     sys.exit()
        # if (self.total_deleted_op_nodes[self.env_idxs, chosen_op]).any():
        #     print(
        #         f'FJSP_Env.py Error from choosing action: Op {chosen_op} is not available')
        #     sys.exit()

        self.step_count += actions.size

        candidate_add_flag = (chosen_op2 != self.job_last_op_id[self.env_idxs, chosen_job2])
        self.candidate[self.env_idxs, chosen_job2] += candidate_add_flag
        self.mask[self.env_idxs, chosen_job2] = (1 - candidate_add_flag)

        # the start processing time of chosen operations
        chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job2],
                                  self.mch_free_time[self.env_idxs, chosen_mch2])

        self.op_ct[self.env_idxs, chosen_op2] = chosen_op_st + self.op_pt[
            self.env_idxs, chosen_op2, chosen_mch2]
        self.candidate_free_time[self.env_idxs, chosen_job2] = self.op_ct[self.env_idxs, chosen_op2]
        self.mch_free_time[self.env_idxs, chosen_mch2] = self.op_ct[self.env_idxs, chosen_op2]

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job2],
                                       self.true_mch_free_time[self.env_idxs, chosen_mch2])
        # self.true_op_st[self.env_idxs, chosen_op] = true_chosen_op_st

        self.true_op_ct[self.env_idxs, chosen_op2] = true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op2, chosen_mch2]
        self.true_candidate_free_time[self.env_idxs, chosen_job2] = self.true_op_ct[
            self.env_idxs, chosen_op2]
        self.true_mch_free_time[self.env_idxs, chosen_mch2] = self.true_op_ct[
            self.env_idxs, chosen_op2]

        candidate_add_flag = (chosen_op2 != self.job_last_op_id[self.env_idxs, chosen_job2])

        self.current_makespan = np.maximum(self.current_makespan, np.max(self.true_op_ct[self.env_idxs, chosen_op2]))

        mask_temp = candidate_add_flag.any(axis=1)
        self.candidate_pt[mask_temp, chosen_job2[candidate_add_flag]] = self.unmasked_op_pt[mask_temp, chosen_op2[candidate_add_flag] + 1]
        self.candidate_process_relation[mask_temp, chosen_job2[candidate_add_flag]] = \
            self.reverse_process_relation[mask_temp, chosen_op2[candidate_add_flag] + 1]
        self.candidate_process_relation[~candidate_add_flag.all(axis=1), chosen_job2[~candidate_add_flag]] = 1


        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        true_candidateFT_for_compare = np.expand_dims(self.true_candidate_free_time, axis=2)
        true_mchFT_for_compare = np.expand_dims(self.true_mch_free_time, axis=1)
        self.true_pair_free_time = np.maximum(true_candidateFT_for_compare, true_mchFT_for_compare)

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)
        true_schedule_matrix = ma.array(self.true_pair_free_time, mask=self.candidate_process_relation)

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data
        
        self.true_next_schedule_time = np.min(
            true_schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        self.remain_process_relation[self.env_idxs, chosen_op2] = 0
        self.op_scheduled_flag[self.env_idxs, chosen_op2] = 1

        self.deleted_op_nodes = np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),self.op_scheduled_flag)
        
        
        if dyn_ver != 0:
            activate_future_op_node = np.minimum(chosen_op2 + self.activate_num_op, self.job_last_op_id[self.env_idxs, chosen_job2])

            
            if dyn_ver != 14:
                invalid_job_mask = torch.tensor(self.candidate_process_relation).all(dim=2) #shape torch.Size([num_enviroments, num_jobs])
                invalid_op_mask = torch.zeros((self.number_of_envs, self.number_of_ops), dtype=torch.bool)
                invalid_op_mask |= torch.any(self.job_op_match_mask * invalid_job_mask.unsqueeze(-1), dim=1)
            
            np.put_along_axis(
                self.deleted_future_op_nodes,
                activate_future_op_node,
                0,
                axis=1
            )

            # Combine deleted nodes
            self.total_deleted_op_nodes = np.logical_or(
                self.deleted_op_nodes,
                self.deleted_future_op_nodes,
            )
            if dyn_ver != 14:
                self.total_deleted_op_nodes = np.logical_or(
                    self.total_deleted_op_nodes,
                    invalid_op_mask.cpu().numpy(),
                )

            # Efficient broadcasting for delete_mask_fea_j
            self.delete_mask_fea_j = np.broadcast_to(
                self.total_deleted_op_nodes[:, :, np.newaxis],
                (self.total_deleted_op_nodes.shape[0], self.total_deleted_op_nodes.shape[1], self.op_fea_dim)
            )
        else:
            self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))
        

        """
            update the state
        """
        pruned_op_mask = self.update_op_mask(dyn_ver)
        
        # update operation raw features
        diff = self.op_ct[self.env_idxs, chosen_op2] - self.op_ct_lb[self.env_idxs, chosen_op2]
        mask1 = (self.op_idx >= chosen_op2[:,:,None]) & (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job2] + 1)[:,:, None])
        self.op_ct_lb[self.env_idxs] += (diff[:, :, None]* mask1).sum(axis=1)

        # mask2 = (self.op_idx[None, :] >= self.job_first_op_id[self.env_idxs, chosen_job2][:, :, None]) & (self.op_idx[None, :] < self.job_last_op_id[self.env_idxs, chosen_job2][:, :, None])
        # # self.op_match_job_left_op_nums[self.env_idxs] -= mask2.sum(axis=1).astype(int)
        # # self.op_match_job_remain_work[self.env_idxs] -= (self.op_mean_pt[self.env_idxs, chosen_op2][:,:,None] * mask2).sum(axis=1)

        # masks_seq = []
        # for a in range(actions.size):
        #     # pruned_op_mask = self.update_op_mask(dyn_ver)
        #     op = np.array([chosen_op2[0,a]])
        #     job = np.array([chosen_job2[0,a]])

        #     """
        #     not same
        #     """
        #     mask3 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, job])[:,
        #                             np.newaxis]) & \
        #             (self.op_idx < (self.job_last_op_id[self.env_idxs, job] + 1)[:,
        #                         np.newaxis])
        #     mask4 = (self.op_idx >= self.job_first_op_id[self.env_idxs, job][:, None]) & (self.op_idx < (self.job_last_op_id[self.env_idxs, job] + 1)[:, None])
        #     masks_seq.append(mask4.astype(int))

        #     self.op_match_job_left_op_nums[mask3] -= 1
        #     # self.op_match_job_left_op_nums[mask2[:,a]] -= 1
        #     # self.op_match_job_left_op_nums[self.env_idxs] -= mask2.sum(axis=1).astype(int)
        #     # self.op_match_job_remain_work[mask2] -= \
        #     #     np.tile(self.op_mean_pt[self.env_idxs, op][:, np.newaxis], (1, self.number_of_ops))[mask2]
        #     # self.op_match_job_remain_work[mask2[:,a]] -= \
        #     #     np.tile(self.op_mean_pt[self.env_idxs, op][:, np.newaxis], (1, self.number_of_ops))[mask2[:,a]]
        #     self.op_match_job_remain_work[mask3] -= \
        #         np.tile(self.op_mean_pt[self.env_idxs, op][:, np.newaxis], (1, self.number_of_ops))[mask3]

        # masks_seq = np.stack(masks_seq, axis=1)

        starts = self.job_first_op_id[self.env_idxs[:, None], chosen_job2]
        ends   = (self.job_last_op_id[self.env_idxs[:, None], chosen_job2] + 1)

        mask4 = (self.op_idx[None, :] >= starts[:, :, None]) & (self.op_idx[None, :] < ends[:, :, None])

        # # mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:,
        # #                         np.newaxis]) & \
        # #         (self.op_idx < (self.true_job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
        # #                        np.newaxis])
        # mask3 = (self.op_idx[None, :] >= self.job_first_op_id[self.env_idxs, chosen_job2][:, :, None]) & (self.op_idx[None, :] < self.job_last_op_id[self.env_idxs, chosen_job2][:, :, None])
        self.op_match_job_left_op_nums[self.env_idxs] -= mask4.sum(axis=1).astype(int)
        # # self.op_match_job_left_op_nums[mask2] -= 1
        self.op_match_job_remain_work[self.env_idxs] -= (self.op_mean_pt[self.env_idxs, chosen_op2][:,:,None] * mask4).sum(axis=1)
        # # self.op_match_job_remain_work[mask2] -= \
        # #     np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[mask2]

        
        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                        - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                        np.expand_dims(self.next_schedule_time, axis=1), 0)
        
        self.construct_op_features()

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = self.true_pair_free_time > self.true_next_schedule_time[:, np.newaxis, np.newaxis]

        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # update comp_idx
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_mch_mask()

        # update machine raw features
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums -= self.process_relation[
            self.env_idxs, chosen_op2].sum(axis=1)
        
        # self.mch_current_available_op_nums -= self.process_relation[
        #     self.env_idxs, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        mch_delete_mask =  np.copy(self.delete_mask_fea_m)
        if dyn_ver == 14 or dyn_ver == 15:
            invalid_mch_mask = torch.tensor(self.candidate_process_relation).all(dim=1) #shape torch.Size([num_enviroments, num_machine])
            mch_delete_mask[mch_free_flag] = 1

        self.construct_mch_features(mch_delete_mask)

        self.construct_pair_features()

        if dyn_ver == 10 or dyn_ver == 12:
            fea_pairs = self.fea_pairs.copy()
            fea_pairs[invalid_job_mask.cpu(), :] = 0
        else:
            fea_pairs = self.fea_pairs
            

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        self.state.update(self.fea_j, pruned_op_mask, self.fea_m, self.mch_mask,
                        self.dynamic_pair_mask, self.comp_idx, self.candidate,
                        fea_pairs)
        # candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
        # self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
        # self.mask[self.env_idxs, chosen_job] = (1 - candidate_add_flag)

        # # the start processing time of chosen operations
        # chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job],
        #                           self.mch_free_time[self.env_idxs, chosen_mch])

        # self.op_ct[self.env_idxs, chosen_op] = chosen_op_st + self.op_pt[
        #     self.env_idxs, chosen_op, chosen_mch]
        # self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
        # # breakpoint()
        # self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]

        # true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],
        #                                self.true_mch_free_time[self.env_idxs, chosen_mch])
        # # self.true_op_st[self.env_idxs, chosen_op] = true_chosen_op_st

        # self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
        #     self.env_idxs, chosen_op, chosen_mch]
        # self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
        #     self.env_idxs, chosen_op]
        # self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
        #     self.env_idxs, chosen_op]
        

        # self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[self.env_idxs, chosen_op])
        # self.current_makespan = np.maximum(self.current_makespan, np.max(self.true_op_ct[self.env_idxs, chosen_op]))
        # update the candidate message
            
        
        # mask_temp = candidate_add_flag.any(axis=1)
        # self.candidate_pt[mask_temp, chosen_job[candidate_add_flag]] = self.unmasked_op_pt[mask_temp, chosen_op[candidate_add_flag] + 1]
        # # breakpoint()
        # self.candidate_process_relation[mask_temp, chosen_job[candidate_add_flag]] = \
        #     self.reverse_process_relation[mask_temp, chosen_op[candidate_add_flag] + 1]
        # self.candidate_process_relation[~candidate_add_flag.all(axis=1), chosen_job[~candidate_add_flag]] = 1

        # compute the next schedule time

        # [E, J, M]
        # candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        # mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        # self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        # schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)

        # self.next_schedule_time = np.min(
        #     schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        # # breakpoint()
        # self.remain_process_relation[self.env_idxs, chosen_op] = 0
        # self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

        """
            update the mask for deleting nodes
        """
        # self.deleted_op_nodes = \
        #     np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
        #                    self.op_scheduled_flag)
       
        # Combine deleted nodes
        # if dyn_ver != 0:
        #     activate_future_op_node = np.minimum(chosen_op + self.activate_num_op, self.job_last_op_id[self.env_idxs, chosen_job])
        #     # breakpoint()
        #     invalid_job_mask = torch.tensor(self.candidate_process_relation).all(dim=2) #shape torch.Size([num_enviroments, num_jobs])
        #     invalid_op_mask = torch.zeros((self.number_of_envs, self.number_of_ops), dtype=torch.bool)
        #     invalid_op_mask |= torch.any(self.job_op_match_mask * invalid_job_mask.unsqueeze(-1), dim=1)
        
        #     np.put_along_axis(
        #         self.deleted_future_op_nodes,
        #         activate_future_op_node,
        #         0,
        #         axis=1
        #     )
        #     self.total_deleted_op_nodes = np.logical_or.reduce(np.stack([self.deleted_op_nodes, self.deleted_future_op_nodes, invalid_op_mask.cpu().numpy()]), axis=0)
        #     self.delete_mask_fea_j = np.broadcast_to(
        #         self.total_deleted_op_nodes[:, :, np.newaxis],
        #         (self.total_deleted_op_nodes.shape[0], self.total_deleted_op_nodes.shape[1], self.op_fea_dim)
        #     )

        # if dyn_ver != 0:
        #     for a in range(actions.size):
        #         self.deleted_op_nodes = np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),self.op_scheduled_flag)

        #         activate_future_op_node = np.minimum(chosen_op[0,a] + self.activate_num_op, self.job_last_op_id[self.env_idxs, chosen_job[0,a]])
        #         # breakpoint()
        #         invalid_job_mask = torch.tensor(self.candidate_process_relation).all(dim=2) #shape torch.Size([num_enviroments, num_jobs])
        #         invalid_op_mask = torch.zeros((self.number_of_envs, self.number_of_ops), dtype=torch.bool)
        #         invalid_op_mask |= torch.any(self.job_op_match_mask * invalid_job_mask.unsqueeze(-1), dim=1)
            
        #         np.put_along_axis(
        #             self.deleted_future_op_nodes,
        #             activate_future_op_node[:, np.newaxis],
        #             0,
        #             axis=1
        #         )
        #         self.total_deleted_op_nodes = np.logical_or.reduce(np.stack([self.deleted_op_nodes, self.deleted_future_op_nodes, invalid_op_mask.cpu().numpy()]), axis=0)
        #         self.delete_mask_fea_j = np.broadcast_to(
        #             self.total_deleted_op_nodes[:, :, np.newaxis],
        #             (self.total_deleted_op_nodes.shape[0], self.total_deleted_op_nodes.shape[1], self.op_fea_dim)
        #         )
        # else:
        #     self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
        #                                 (1, 1, self.op_fea_dim))
        
        

        """
            update the state
        """
        # pruned_op_mask = self.update_op_mask(dyn_ver)

        # update operation raw features
        # for a in range(actions.size):
        #     pruned_op_mask = self.update_op_mask(dyn_ver)
        #     op = np.array([chosen_op[0,a]])
        #     job = np.array([chosen_job[0,a]])
        #     diff = self.op_ct[self.env_idxs, op] - self.op_ct_lb[self.env_idxs, op]

        #     mask1 = (self.op_idx >= op[:, np.newaxis]) & \
        #             (self.op_idx < (self.job_last_op_id[self.env_idxs, job] + 1)[:,
        #                         np.newaxis])
        #     self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]

        #     mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, job])[:,
        #                             np.newaxis]) & \
        #             (self.op_idx < (self.job_last_op_id[self.env_idxs, job] + 1)[:,
        #                         np.newaxis])
        #     self.op_match_job_left_op_nums[mask2] -= 1
        #     self.op_match_job_remain_work[mask2] -= \
        #         np.tile(self.op_mean_pt[self.env_idxs, op][:, np.newaxis], (1, self.number_of_ops))[mask2]
            
        #     self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        #     self.op_waiting_time[self.env_job_idx, self.candidate] = \
        #         (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
        #                                     - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
        #             self.env_job_idx, self.candidate]

        #     self.op_remain_work = np.maximum(self.op_ct -
        #                                     np.expand_dims(self.next_schedule_time, axis=1), 0)
            
        #     # breakpoint()
        #     self.construct_op_features()

        #     # update dynamic pair mask
        #     self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        #     self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

        #     self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        #     # update comp_idx
        #     self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        #     self.update_mch_mask()

        #     # update machine raw features
        #     self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        #     self.mch_current_available_op_nums -= self.process_relation[
        #         self.env_idxs, chosen_op].sum(axis=1)

        #     mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        #     mch_free_flag = mch_free_duration < 0
        #     self.mch_working_flag = mch_free_flag + 0
        #     self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

        #     self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        #     mch_delete_mask =  np.copy(self.delete_mask_fea_m)
        #     if dyn_ver == 14 or dyn_ver == 15:
        #         invalid_mch_mask = torch.tensor(self.candidate_process_relation).all(dim=1) #shape torch.Size([num_enviroments, num_machine])
        #         mch_delete_mask[mch_free_flag] = 1

        #     self.construct_mch_features(mch_delete_mask)

        #     self.construct_pair_features()

        #     if dyn_ver == 10 or dyn_ver == 12:
        #         fea_pairs = self.fea_pairs.copy()
        #         fea_pairs[invalid_job_mask.cpu(), :] = 0
        #     else:
        #         fea_pairs = self.fea_pairs
                

        #     # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        #     reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        #     self.max_endTime = np.max(self.op_ct_lb, axis=1)
            
        # diff = self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]

        # # mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & \
        # #         (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
        # #                        np.newaxis])
        # mask1 = (self.op_idx >= chosen_op[:,:,None]) & (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,:, None])
        # # self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]
        # self.op_ct_lb[self.env_idxs] += (diff[:, :, None]* mask1).sum(axis=1)

        # # mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:,
        # #                         np.newaxis]) & \
        # #         (self.op_idx < (self.true_job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
        # #                        np.newaxis])
        # mask2 = (self.op_idx[None, :] >= self.job_first_op_id[self.env_idxs, chosen_job][:, :, None]) & (self.op_idx[None, :] < self.job_last_op_id[self.env_idxs, chosen_job][:, :, None])
        # self.op_match_job_left_op_nums[self.env_idxs] -= mask2.sum(axis=1).astype(int)
        # # self.op_match_job_left_op_nums[mask2] -= 1
        # self.op_match_job_remain_work[self.env_idxs] -= (self.op_mean_pt[self.env_idxs, chosen_op][:,:,None] * mask2).sum(axis=1)
        # # self.op_match_job_remain_work[mask2] -= \
        # #     np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[mask2]

        # self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        # self.op_waiting_time[self.env_job_idx, self.candidate] = \
        #     (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
        #                                  - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
        #         self.env_job_idx, self.candidate]

        # self.op_remain_work = np.maximum(self.op_ct -
        #                                  np.expand_dims(self.next_schedule_time, axis=1), 0)
        
        # # breakpoint()
        # self.construct_op_features()

        # # update dynamic pair mask
        # self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        # self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

        # self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # # update comp_idx
        # self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        # self.update_mch_mask()

        # # update machine raw features
        # self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        # self.mch_current_available_op_nums -= self.process_relation[
        #     self.env_idxs, chosen_op].sum(axis=1)

        # mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        # mch_free_flag = mch_free_duration < 0
        # self.mch_working_flag = mch_free_flag + 0
        # self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

        # self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        # mch_delete_mask =  np.copy(self.delete_mask_fea_m)
        # if dyn_ver == 14 or dyn_ver == 15:
        #     invalid_mch_mask = torch.tensor(self.candidate_process_relation).all(dim=1) #shape torch.Size([num_enviroments, num_machine])
        #     mch_delete_mask[mch_free_flag] = 1

        # self.construct_mch_features(mch_delete_mask)

        # self.construct_pair_features()

        # if dyn_ver == 10 or dyn_ver == 12:
        #     fea_pairs = self.fea_pairs.copy()
        #     fea_pairs[invalid_job_mask.cpu(), :] = 0
        # else:
        #     fea_pairs = self.fea_pairs
            

        # # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        # reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        # self.max_endTime = np.max(self.op_ct_lb, axis=1)

        # update the state
        # self.state.update(self.fea_j, pruned_op_mask, self.fea_m, self.mch_mask,
        #                   self.dynamic_pair_mask, self.comp_idx, self.candidate,
        #                   fea_pairs)

        return self.state, np.array(reward), self.done()

    def done(self):
        """
            compute the done flag
        """
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_ops)

    def construct_op_features(self):
        """
            construct operation raw features
        """
        self.fea_j = np.stack((self.op_scheduled_flag,
                               self.op_ct_lb,
                               self.op_min_pt,
                               self.pt_span,
                               self.op_mean_pt,
                               self.op_waiting_time,
                               self.op_remain_work, #Remaining processing time:
                               self.op_match_job_left_op_nums, #Job remaining number of operations
                               self.op_match_job_remain_work, #Job remaining workload
                               self.op_available_mch_nums), axis=2)

        if self.step_count != self.number_of_ops:
            self.norm_op_features()

    def norm_op_features(self):
        """
            normalize operation raw features (across the second dimension)
        """
        self.fea_j[self.delete_mask_fea_j] = 0
        num_delete_nodes = np.count_nonzero(self.total_deleted_op_nodes, axis=1)
        num_left_nodes = self.number_of_ops - num_delete_nodes[:, np.newaxis]
        
        # mean_fea_j = np.sum(self.fea_j, axis=1, where=~self.total_deleted_op_nodes[:, :, np.newaxis]) / num_left_nodes
        # var_fea_j = np.var(self.fea_j, axis=1, where=~self.total_deleted_op_nodes[:, :, np.newaxis])
        # std_fea_j = np.sqrt(var_fea_j * self.number_of_ops / num_left_nodes)
        # self.fea_j = np.where( ~self.delete_mask_fea_j,  (self.fea_j - mean_fea_j[:, np.newaxis, :]) / (std_fea_j[:, np.newaxis, :] + 1e-8), 0)

        mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes
        temp = np.where(self.delete_mask_fea_j, mean_fea_j[:, np.newaxis, :], self.fea_j)
        var_fea_j = np.var(temp, axis=1)

        std_fea_j = np.sqrt(var_fea_j * self.number_of_ops / num_left_nodes)

        self.fea_j = (temp - mean_fea_j[:, np.newaxis, :]) / \
                     (std_fea_j[:, np.newaxis, :] + 1e-8)

    def construct_mch_features(self, mch_delete_mask):
        """
            construct machine raw features
        """
        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_min_pt,
                               self.mch_mean_pt,
                               self.mch_waiting_time,
                               self.mch_remain_work,
                               self.mch_free_time,
                               self.mch_working_flag), axis=2)

        if self.step_count != self.number_of_ops:
            self.norm_machine_features(mch_delete_mask)

    def norm_machine_features(self, mch_delete_mask):
        """
            normalize machine raw features (across the second dimension)
        """
        self.fea_m[mch_delete_mask] = 0
        num_delete_mchs = np.count_nonzero(mch_delete_mask[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = self.number_of_machines - num_delete_mchs
        mean_fea_m = np.sum(self.fea_m, axis=1) / num_left_mchs
        temp = np.where(mch_delete_mask,
                        mean_fea_m[:, np.newaxis, :], self.fea_m)
        var_fea_m = np.var(temp, axis=1)
        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        self.fea_m = (temp - mean_fea_m[:, np.newaxis, :]) / \
                     (std_fea_m[:, np.newaxis, :] + 1e-8)

    def construct_pair_features(self):
        """
            construct pair features
        """
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)

        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True) \
            .filled(0 + 1e-8)

        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True). \
            filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8

        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :,
                         np.newaxis] + self.mch_waiting_time[:, np.newaxis, :]

        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work
                                                [self.env_job_idx, self.candidate],
                                                axis=-1) + 1e-8

        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_op_max_pt,
                                   self.candidate_pt / mch_max_candidate_pt,
                                   self.candidate_pt / max_remain_op_pt,
                                   self.candidate_pt / mch_max_remain_op_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_job_remain_work,
                                   pair_wait_time), axis=-1)

    def update_mch_mask(self):
        """
            update 'mch_mask'
        """
        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):
        """
            initialize 'mch_mask'
        """
        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):
        """
            initialize 'op_mask'
        """
        self.op_mask = np.full(shape=(self.number_of_envs, self.number_of_ops, 3),
                               fill_value=0, dtype=np.float32)
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def update_op_mask(self, dyn_ver):
        """
            update 'op_mask'
        """
        object_mask = np.zeros_like(self.op_mask)
        
        if dyn_ver == 10 or dyn_ver == 11:
            object_mask[:, :, 2] = self.total_deleted_op_nodes
            object_mask[:, 1:, 0] = self.total_deleted_op_nodes[:, :-1]
            object_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)
        else:
            object_mask[:, :, 2] = self.deleted_op_nodes
            object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
            object_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)
            

        return object_mask


    def logic_operator(self, x, flagT=True):
        """
            a customized operator for computing some masks
        :param x: a 3-d array with shape [s,a,b]
        :param flagT: whether transpose x in the last two dimensions
        :return:  a 4-d array c, where c[i,j,k,l] = x[i,j,l] & x[i,k,l] for each i,j,k,l
        """
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)
