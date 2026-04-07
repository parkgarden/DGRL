import bisect
import random
from env.utils.dyn_mach_job_op_BA2 import *
from env.utils.generator import *
from env.utils.dyn_graph_BA2 import Graph
import torch
import time
from collections import defaultdict

class JSP_Instance:
    def __init__(self, args):
        self.args, self.process_time_range = args, [1, args.max_process_time]

        self.job_num, self.machine_num = 0, 0

        self.jobs, self.machines = [], []
        self.time_stamp = []
        self.current_time = 0
        self.max_process_time = 0
        self.assigned_nodes = []

        self.graph = None

    def generate_case(self):
        self.insert_jobs(job_num=self.job_num)
        
    def insert_jobs(self, job_num):
        self.register_time(0)
        if self.args.instance_type == 'FJSP':
            for i in range(job_num):
                job_id = len(self.jobs)
                if self.args.data_char == 0:
                    op_config = gen_operations_FJSP(self.machine_num, self.process_time_range)
                else:
                    op_config = gen_operations_FJSP_SD2(self.machine_num, self.process_time_range)
                self.jobs.append(Job(args=self.args, job_id=job_id, op_config=op_config))

        elif self.args.instance_type == 'JSP':
            for i in range(job_num):
                job_id = len(self.jobs)
                op_config = gen_operations_JSP(self.machine_num, self.machine_num, self.process_time_range)
                self.jobs.append(Job(args=self.args, job_id=job_id, op_config=op_config))

        # build up the graph
        self.graph = Graph(self.args, self.job_num, self.machine_num)
        for i in range(job_num):
            self.graph.add_job(self.jobs[i])
        
    def reset(self):
        self.job_num = random.randint(3, self.args.data_size)
        self.machine_num = random.randint(3, self.args.data_size)

        self.jobs = []
        self.machines = [Machine(machine_id) for machine_id in range(self.machine_num)]
        self.current_time = 0
        self.time_stamp = []
        self.generate_case()

    def load_instance(self, filename):
        self.jobs = []
        self.current_time = 0
        self.time_stamp = []
        self.assigned_nodes = []
        
        f = open(filename)
        line = f.readline()
        while line[0] == '#':
            line = f.readline()
            
        line = line.split()
        self.job_num, self.machine_num = int(line[0]), int(line[1])
        self.machines = [Machine(machine_id) for machine_id in range(self.machine_num)]

        if self.args.instance_type == "JSP":
            for i in range(self.job_num):
                op_config = []
                line = f.readline().split()
                for j in range(self.machine_num):
                    machine_id, process_time = int(line[j * 2]), int(line[j * 2 + 1])
                    if process_time == 0:
                        continue
                    machine_and_processtime = [(machine_id, process_time)]
                    op_config.append({"id": j, "machine_and_processtime": machine_and_processtime})
                self.jobs.append(Job(args=self.args, job_id=i, op_config=op_config))
        
        if self.args.instance_type == "FJSP":
            for i in range(self.job_num):
                op_config = []
                line = f.readline().split()
                op_num = int(line[0])
                cur = 1
                for j in range(op_num):
                    machine_and_processtime = []
                    machine_num = int(line[cur])
                    cur += 1
                    for _ in range(machine_num):
                        machine_id, process_time = int(line[cur]), int(line[cur + 1])
                        machine_and_processtime.append((machine_id - 1, process_time))
                        cur += 2
                    op_config.append({"id": j, "machine_and_processtime": machine_and_processtime})
                self.jobs.append(Job(self.job_num, args=self.args, job_id=i, op_config=op_config))

        self.graph = Graph(self.args, self.job_num, self.machine_num)
        for i in range(self.job_num):
            self.graph.add_job(self.jobs[i])
        
        self.register_time(0)

    def done(self):
        for job in self.jobs:
            if job.done() == False:
                return False
        return True
    
    def get_graph_data(self):
        self.graph.update_feature(self.jobs, self.machines, self.current_time)
        data, op_unfinished = self.graph.get_data()
        return data.to(self.args.device), op_unfinished
    
        
    def assign(self, step_ops):
        n = len(step_ops)
        # if not isinstance(step_ops, list):
        #     breakpoint()
        #     step_ops = [step_ops]
        # # breakpoint()
        try:
            job_ids = torch.tensor([int(s['job_id']) for s in step_ops], dtype=torch.long)  # (n,)
        except Exception:
            breakpoint()
        op_ids  = torch.tensor([int(s['op_id'])  for s in step_ops], dtype=torch.long)  # (n,)
        m_ids   = torch.tensor([int(s['m_id'])    for s in step_ops], dtype=torch.long)  # (n,)
        p_times = torch.tensor([float(s['process_time']) for s in step_ops], dtype=torch.float32)  # (n,)

        avail_times_list = [self.jobs[int(j)].current_op().avai_time for j in job_ids.tolist()]
        avail_times = torch.tensor(avail_times_list, dtype=torch.float32)
        current_time_tensor = torch.tensor(float(self.current_time), dtype=torch.float32)
        start_times = torch.maximum(current_time_tensor, avail_times)

        # job_id, op_id = step_op['job_id'], step_op['op_id']
        op_info_list = []
        for i in range(n):
            op_info_list.append({
                "job_id"       : int(job_ids[i].item()),
                "op_id"        : int(op_ids[i].item()),
                "current_time" : float(start_times[i].item()),
                "process_time" : float(p_times[i].item())
            })

        mas = m_ids.reshape(n, 1).long()

        for mid in m_ids.tolist():
            self.machines[int(mid)].occupied = 1
        
        job_to_selected_ops = defaultdict(list)
        for j, op in zip(job_ids.tolist(), op_ids.tolist()):
            job_to_selected_ops[int(j)].append(int(op))

        for j, sel_ops in job_to_selected_ops.items():
            job = self.jobs[j]
            sel_set = set(sel_ops)
            # mark every op except those selected as occupied
            for i_op in range(len(job.operations)):
                if i_op not in sel_set:
                    job.operations[i_op].occupied = 1

        finished_times = torch.empty(n, dtype=torch.float32)
        assigned_nodes = []
        removals = []
        for i, info in enumerate(op_info_list):
            j = info['job_id']; op = info['op_id']; m = int(m_ids[i].item())

            # preserve assigned_nodes as original
            assigned_nodes.append(self.jobs[j].operations[op].node_id)

            # Machine processes the operation: keep per-action call
            op_finished_time = self.machines[m].process_op(info)
            finished_times[i] = float(op_finished_time)

            # update job's current op (same as original)
            self.jobs[j].current_op().update(self.current_time, info['process_time'])

            # update next op availability if there is a next op
            if self.jobs[j].next_op() != -1:
                self.jobs[j].update_current_op(avai_time=op_finished_time)

            # delete node in graph if requested (keeps same behavior)
            if self.args.delete_node:
                removals.append((j, self.jobs[j].operations[op]))
                # self.graph.remove_node(j, self.jobs[j].operations[op])

        self.assigned_nodes.extend(assigned_nodes)
        # breakpoint()
        new_times = sorted(set([t for t in finished_times.tolist()]))

        old_ts = self.time_stamp
        i = j = 0
        merged = []
        while i < len(old_ts) and j < len(new_times):
            a, b = old_ts[i], new_times[j]
            if a < b:
                merged.append(a); i += 1
            elif a > b:
                merged.append(b); j += 1
            else:
                merged.append(a); i += 1; j += 1
        # append rest
        if i < len(old_ts):
            merged.extend(old_ts[i:])
        if j < len(new_times):
            merged.extend(new_times[j:])
        self.time_stamp = merged

        if self.args.delete_node:
            # removals.append((j, self.jobs[j].operations[op]))
            self.graph.remove_nodes(removals)


        if self.args.dyn != 0:
            # remove distant operation
            if self.args.graph_len == 0:
                graph_len = self.machine_num
            else:
                graph_len = self.args.graph_len
            for j, sel_ops in job_to_selected_ops.items():
                chosen_op = min(sel_ops)
                activate_future_op_node = min(chosen_op + graph_len, self.jobs[j].op_num - 1)
                self.jobs[j].operations[activate_future_op_node].distant = 0

    def register_time(self, time):
        index = bisect.bisect_left(self.time_stamp, time)
        if index == len(self.time_stamp) or self.time_stamp[index] != time:
            self.time_stamp.insert(index, time)
    
    def update_time(self):
        self.current_time = self.time_stamp.pop(0)
    
    def current_avai_ops(self):
        if self.done() == True:
            return None

        avai_ops = []
        avai_mat = np.zeros((self.machine_num, self.job_num),)

        for m in self.machines:
            if m.avai_time() > self.current_time:
                continue
            for job in self.jobs:
                if job.done() == True or job.current_op().avai_time > self.current_time:
                    continue
                # unmask the operation that finished processing
                # breakpoint()
                for i in range(job.current_op().op_id, job.op_num):
                    job.operations[i].occupied = 0
                # job.current_op().occupied = 0

                for machine_and_processtime in job.current_op().machine_and_processtime:
                    if m.machine_id == machine_and_processtime[0]:
                        avai_mat[machine_and_processtime[0]][job.job_id] = machine_and_processtime[1]
            
            if self.args.dyn != 0:
                #remove complete machine
                if ~torch.isin(self.graph.m_edge_idx, m.machine_id).any():
                    m.complete = 1
                # unmask the machine that finished processing
                m.occupied = 0


        for i in range(self.job_num):
            avai_m_idx = np.nonzero(avai_mat[:,i])
            # if there is only one available machine for the job i and only one compatible job for the machine, then assign the pair  
            if len(avai_m_idx) == 1 and np.count_nonzero(avai_mat[avai_m_idx[0]]) == 1: # 1<->1 op<->m, it must be assigned
                # breakpoint()
                self.assign([{
                    'm_id'          : avai_m_idx[0].item(),
                    'job_id'        : i,
                    'op_id'         : self.jobs[i].current_op_id,
                    'node_id'       : self.jobs[i].current_op().node_id,
                    'process_time'  : avai_mat[avai_m_idx[0].item()][i]
                }])
                avai_mat[avai_m_idx[0].item()][i] = 0
        
        candidates = np.where(avai_mat > 0)
        for i in range(len(candidates[0])):
            m_id   = candidates[0][i]
            job_id = candidates[1][i]
            process_time = avai_mat[m_id][job_id]
            
            avai_ops.append({
                'm_id'          : m_id,
                'job_id'        : job_id,
                'op_id'         : self.jobs[job_id].current_op_id,
                'node_id'       : self.jobs[job_id].current_op().node_id,
                'process_time'  : process_time
            })
        if len(avai_ops) == 0:
            self.update_time()
            return self.current_avai_ops()
        else:
            return avai_ops
        
    def get_max_process_time(self):
        return self.graph.get_max_process_time()
