import torch
from torch_geometric.data import HeteroData
import numpy as np
import time
from bisect import bisect_left
from collections import Counter

AVAILABLE = 0
PROCESSED = 1
COMPLETE = 3
FUTURE = 2

def binary_search(list_, target):
    left, right = 0, len(list_)
    pos = bisect_left(list_, target, left, right)
    return pos if pos != right and list_[pos] == target else -1

class Graph:
    def __init__(self, args, job_num, machine_num):

        self.op_op_edge_src_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        self.op_op_edge_tar_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        self.op_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                           # for op<->m
        self.m_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                            # for op<->m
        self.m_m_edge_idx = torch.tensor([[i for i in range(machine_num)]], dtype=torch.int64)  # for m<->m
        self.edge_x = torch.empty(size=(1,0), dtype=torch.int64)

        self.op_x = []
        self.m_x = []

        # self.temp_op_op_edge_src_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        # self.temp_op_op_edge_tar_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        # self.temp_op_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                           # for op<->m
        # self.temp_m_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                            # for op<->m
        # self.temp_m_m_edge_idx = torch.tensor([[i for i in range(machine_num)]], dtype=torch.int64)  # for m<->m
        # self.temp_edge_x = torch.empty(size=(1,0), dtype=torch.int64)

        # self.temp_op_x = []
        # self.temp_m_x = []

        self.args = args
        self.job_num = job_num
        self.machine_num = machine_num
        self.op_num = 0
        self.op_unfinished = []
        self.current_op = [] 

        self.max_process_time = 0.

    # def temp_graph_reset(self):
    #     self.temp_op_op_edge_src_idx =  self.op_op_edge_src_idx.clone().detach()
    #     self.temp_op_op_edge_tar_idx = self.op_op_edge_tar_idx.clone().detach()                    # for op<->op
    #     self.temp_op_edge_idx = self.op_edge_idx.clone().detach()                           # for op<->m
    #     self.temp_m_edge_idx = self.m_edge_idx.clone().detach()                            # for op<->m
    #     self.temp_m_m_edge_idx = self.m_m_edge_idx.clone().detach()  # for m<->m
    #     self.temp_edge_x = self.edge_x.clone().detach()

    #     self.temp_op_x = self.op_x.clone().detach()
    #     self.temp_m_x = self.m_x.clone().detach()


    def get_data(self):
        data = HeteroData()

        data['op'].x    = torch.FloatTensor(self.op_x)
        data['m'].x     = torch.FloatTensor(self.m_x)

        data['op', 'to', 'op'].edge_index   = torch.cat((self.op_op_edge_src_idx, self.op_op_edge_tar_idx), dim=0).contiguous()
        data['op', 'to', 'm'].edge_index    = torch.cat((self.op_edge_idx, self.m_edge_idx), dim=0).contiguous()
        data['m', 'to', 'op'].edge_index    = torch.cat((self.m_edge_idx, self.op_edge_idx), dim=0).contiguous()
        data['m', 'to', 'm'].edge_index     = torch.cat((self.m_m_edge_idx, self.m_m_edge_idx), dim=0).contiguous()

        return data, self.op_unfinished
    
    # def get_temp_data(self):
    #     data = HeteroData()

    #     data['op'].x    = torch.FloatTensor(self.temp_op_x)
    #     data['m'].x     = torch.FloatTensor(self.temp_m_x)

    #     data['op', 'to', 'op'].edge_index   = torch.cat((self.temp_op_op_edge_src_idx, self.temp_op_op_edge_tar_idx), dim=0).contiguous()
    #     data['op', 'to', 'm'].edge_index    = torch.cat((self.temp_op_edge_idx, self.temp_m_edge_idx), dim=0).contiguous()
    #     data['m', 'to', 'op'].edge_index    = torch.cat((self.temp_m_edge_idx, self.temp_op_edge_idx), dim=0).contiguous()
    #     data['m', 'to', 'm'].edge_index     = torch.cat((self.temp_m_m_edge_idx, self.temp_m_m_edge_idx), dim=0).contiguous()

    #     return data, self.op_unfinished
       
    def add_job(self, job):
        src, tar = self.fully_connect(self.op_num, job.op_num)
        self.op_op_edge_src_idx = torch.cat((self.op_op_edge_src_idx, src.unsqueeze(0)), dim=1)
        self.op_op_edge_tar_idx = torch.cat((self.op_op_edge_tar_idx, tar.unsqueeze(0)), dim=1)
        self.current_op.append(0)
        
        for i in range(job.op_num):
            job.operations[i].node_id = self.op_num # set index of an op in the graph
            op = job.operations[i]
            self.op_edge_idx    = torch.cat((self.op_edge_idx,  torch.tensor([[self.op_num for _ in range(len(op.machine_and_processtime))]])), dim=1)
            self.m_edge_idx     = torch.cat((self.m_edge_idx,   torch.tensor([[machine_and_processtime[0] for machine_and_processtime in op.machine_and_processtime]])), dim=1)
            self.edge_x         = torch.cat((self.edge_x,       torch.tensor([[machine_and_processtime[1] for machine_and_processtime in op.machine_and_processtime]])), dim=1)

            self.op_unfinished.append(self.op_num)
            self.op_num += 1

    def update_feature(self, jobs, machines, current_time):
        self.op_x, self.m_x = [], []
        self.max_process_time = self.get_max_process_time()
        # op feature
        # [idle(bin), processed(bin), exp process time, waiting time, remaining job]
        for i in range(len(jobs)):
            for j in range(self.current_op[i], len(jobs[i].operations)):
                op = jobs[i].operations[j]
                status = op.get_status(current_time)
                if self.args.delete_node == True:
                    feat = [0] * 2
                    feat[status // 2] = 1
                else:
                    feat = [0] * 4
                    feat[status] = 1

                
                if op.occupied != 1 and op.distant != 1:
                    # exp process time
                    feat.append(op.expected_process_time / self.max_process_time)

                    # waiting time
                    if status == AVAILABLE:
                        feat.append((current_time - op.avai_time) / self.max_process_time)
                    else:
                        feat.append(0)

                    # remaining job
                    feat.append(jobs[i].acc_expected_process_time[op.op_id] / jobs[i].acc_expected_process_time[0])
                else: # filtering out irrelavant operations
                    feat = [0] * 5

                self.op_x.append(feat)

        
        # machine feature
        # [idle(bin), processing(bin), time to comlete, waiting time]
        for m in machines:
            if m.complete != 1 and m.occupied != 1:
                feat = [0] * 2
                status = m.get_status(current_time)
                feat[status] = 1
                
                if status == AVAILABLE:
                    feat.append(0)
                    feat.append((current_time - m.avai_time()) / self.max_process_time)
                else:
                    feat.append((m.avai_time() - current_time) / self.max_process_time)
                    feat.append(0)
            else: # filtering out unavailable machine
                feat = [0] * 4

            self.m_x.append(feat)
        
    def remove_node(self, job_id, remove_op):
        idx = binary_search(self.op_unfinished, remove_op.node_id)
        self.op_unfinished.pop(idx)
        self.current_op[job_id] += 1

        # op-op
        src_remove_idxs = torch.where(self.op_op_edge_src_idx == idx)[1]
        tar_remove_idxs = torch.where(self.op_op_edge_tar_idx == idx)[1]

        mask = torch.ones(self.op_op_edge_src_idx.shape[1], dtype=bool)
        mask[src_remove_idxs] = False
        mask[tar_remove_idxs] = False

        self.op_op_edge_src_idx = self.op_op_edge_src_idx[:, mask]
        self.op_op_edge_tar_idx = self.op_op_edge_tar_idx[:, mask]

        #op-m, m-op
        remove_idxs = torch.where(self.op_edge_idx == idx)[1]

        mask = torch.ones(self.op_edge_idx.shape[1], dtype=bool)
        mask[remove_idxs] = False

        self.op_edge_idx    = self.op_edge_idx[:, mask]
        self.m_edge_idx     = self.m_edge_idx[:, mask]
        self.edge_x         = self.edge_x[:, mask]

        _, self.op_edge_idx         = torch.unique(self.op_edge_idx, return_inverse=True)
        _, self.op_op_edge_src_idx  = torch.unique(self.op_op_edge_src_idx, return_inverse=True)
        _, self.op_op_edge_tar_idx  = torch.unique(self.op_op_edge_tar_idx, return_inverse=True)
        
        self.op_num -= 1

    def remove_nodes(self, removals):
         # --- 1) Normalize input into two lists: job_ids_list, node_ids_list ---
        job_ids_list = []
        node_ids_list = []
        for item in removals:
            # handle tuple (job_id, remove_op) or dict-like with 'node_id'
            if isinstance(item, (tuple, list)) and len(item) == 2:
                job_id, remove_op = item
                job_ids_list.append(int(job_id))
                # remove_op might be object with node_id attribute or direct node_id int
                node_ids_list.append(int(getattr(remove_op, "node_id", remove_op)))
            else:
                # if single node_id or object passed
                if hasattr(item, "node_id"):
                    node_ids_list.append(int(item.node_id))
                    # job_id not provided: put -1 (we won't update current_op)
                    job_ids_list.append(-1)
                else:
                    # treat item as node_id
                    node_ids_list.append(int(item))
                    job_ids_list.append(-1)

        if len(node_ids_list) == 0:
            return

        # convert to python lists & unique them (we remove each node exactly once)
        node_ids_list = list(dict.fromkeys(node_ids_list))   # preserve order, unique
        # build job increment counts for proper current_op updates
        job_increments = Counter(j for j in job_ids_list if j != -1)

        # --- 2) Map node_ids to indices in self.op_unfinished (same as single remove's binary_search) ---
        # self.op_unfinished is a Python list of node_ids in a specific order.
        # Build a dict mapping node_id -> index for fast lookup.
        op_unfinished_list = self.op_unfinished  # Python list
        id_to_idx = {node_id: idx for idx, node_id in enumerate(op_unfinished_list)}

        # gather remove_idxs (indices in op_unfinished) for each node_id we will drop
        remove_idxs = []
        for nid in node_ids_list:
            if nid not in id_to_idx:
                # node already removed or not found; ignore (or raise if you prefer)
                continue
            remove_idxs.append(int(id_to_idx[nid]))
        if len(remove_idxs) == 0:
            return  # nothing to remove

        remove_idxs_tensor = torch.tensor(remove_idxs, dtype=torch.long)

        # --- 3) Build boolean masks for edge-column pruning (vectorized) ---
        # For op-op edges:
        # op_op_edge_src_idx and op_op_edge_tar_idx compare against op indices (the same idx space
        # used for op_unfinished). We want to drop columns where either src OR tar equals any remove idx.
        if hasattr(torch, "isin"):
            src_match = torch.isin(self.op_op_edge_src_idx, remove_idxs_tensor)
            tar_match = torch.isin(self.op_op_edge_tar_idx, remove_idxs_tensor)
        else:
            # fallback for older torch: build set and compare
            remove_set = set(remove_idxs)
            src_match = torch.zeros_like(self.op_op_edge_src_idx, dtype=torch.bool)
            tar_match = torch.zeros_like(self.op_op_edge_tar_idx, dtype=torch.bool)
            # compare elementwise (vectorized via broadcasting of compare for each remove idx)
            # this fallback still avoids Python per-column loops but loops over remove_idxs
            for r in remove_idxs:
                src_match |= (self.op_op_edge_src_idx == r)
                tar_match |= (self.op_op_edge_tar_idx == r)

        # any column that has a True in either src_match or tar_match should be removed
        cols_to_keep_mask_opop = ~(src_match.any(dim=0) | tar_match.any(dim=0))  # shape: (num_cols_opop,)

        # For op-edge (op-m, m-op) edges: self.op_edge_idx holds op indices in the same op_unfinished index space
        if hasattr(torch, "isin"):
            opedge_match = torch.isin(self.op_edge_idx, remove_idxs_tensor)
        else:
            opedge_match = torch.zeros_like(self.op_edge_idx, dtype=torch.bool)
            for r in remove_idxs:
                opedge_match |= (self.op_edge_idx == r)

        cols_to_keep_mask_opedge = ~opedge_match.any(dim=0)  # shape: (num_cols_opedge,)

        # Use the masks to filter columns on all related tensors
        self.op_op_edge_src_idx = self.op_op_edge_src_idx[:, cols_to_keep_mask_opop]
        self.op_op_edge_tar_idx = self.op_op_edge_tar_idx[:, cols_to_keep_mask_opop]

        self.op_edge_idx = self.op_edge_idx[:, cols_to_keep_mask_opedge]
        self.m_edge_idx  = self.m_edge_idx[:, cols_to_keep_mask_opedge]
        self.edge_x      = self.edge_x[:, cols_to_keep_mask_opedge]

        # --- 4) Re-index / compact indices exactly as you did in single remove_node ---
        # Your original code used torch.unique(..., return_inverse=True) to rebuild index mappings.
        # Do that again for op_edge_idx and op-op edge arrays:
        # Note: unique returns (unique_vals, inverse_indices). The code you had did:
        #    _, self.op_edge_idx         = torch.unique(self.op_edge_idx, return_inverse=True)
        # which effectively flattens and reindexes.
        _, new_op_edge_idx = torch.unique(self.op_edge_idx, return_inverse=True)
        _, new_op_op_src_idx = torch.unique(self.op_op_edge_src_idx, return_inverse=True)
        _, new_op_op_tar_idx = torch.unique(self.op_op_edge_tar_idx, return_inverse=True)

        # Infer original intended shapes: your single-remove assigned the returned 'inverse' into the same variable,
        # so reshape as two-row (or original first-dim) by num_cols. We must restore the original first-dimension.
        # Assume original first-dim sizes:
        op_edge_firstdim = self.op_edge_idx.shape[0]    # typically 2 (src,tar) or feature dims
        opop_firstdim = self.op_op_edge_src_idx.shape[0]
        # Now replace with the reindexed tensors (we need to reshape 'new...' to match original firstdim)
        # The returned 'new_op_edge_idx' is a 1-D tensor of length (num_cols_opedge * firstdim). We'll reshape:
        try:
            self.op_edge_idx = new_op_edge_idx.reshape(op_edge_firstdim, -1)
        except RuntimeError:
            # If shapes don't align, fallback to preserving the current op_edge_idx (already pruned)
            # but at least set it to contiguous
            self.op_edge_idx = self.op_edge_idx.contiguous()

        try:
            self.op_op_edge_src_idx = new_op_op_src_idx.reshape(opop_firstdim, -1)
            self.op_op_edge_tar_idx = new_op_op_tar_idx.reshape(opop_firstdim, -1)
        except RuntimeError:
            self.op_op_edge_src_idx = self.op_op_edge_src_idx.contiguous()
            self.op_op_edge_tar_idx = self.op_op_edge_tar_idx.contiguous()

        # --- 5) Update Python lists: self.op_unfinished and self.current_op per-job counters ---
        remove_node_set = set(node_ids_list)
        self.op_unfinished = [nid for nid in op_unfinished_list if nid not in remove_node_set]

        # increment current_op counters for jobs that lost an op
        for job_id, inc in job_increments.items():
            # ensure job_id is valid
            if 0 <= job_id < len(self.current_op):
                self.current_op[job_id] += inc

        # --- 6) Update op_num ---
        self.op_num = max(0, self.op_num - len(remove_idxs))


    def fully_connect(self, begin, size):
        adj_matrix = torch.ones((size, size))
        idxs = torch.where(adj_matrix > 0)
        return idxs[0] + begin, idxs[1] + begin

    def get_max_process_time(self):
        return torch.max(self.edge_x).item()