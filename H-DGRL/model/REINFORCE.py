import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch.distributions import Categorical
from itertools import accumulate
from model.gnn import GNN
torch.set_printoptions(precision=10)

class REINFORCE(nn.Module):
    def __init__(self, args):
        super(REINFORCE, self).__init__()
        self.args = args
        self.policy_num_layers = args.policy_num_layers
        self.hidden_dim = args.hidden_dim
        self.gnn = GNN(args)
        self.layers = torch.nn.ModuleList() # policy network
        self.layers.append(nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim))
        for _ in range(self.policy_num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, 1))
        
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.baselines = []
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, avai_ops, data, op_unfinished, max_process_time, greedy=False, mode=0, alpha =0):
        # breakpoint()
        x_dict = self.gnn(data)

        if mode == 0:
            score = torch.empty(size=(0, self.args.hidden_dim * 2 + 1)).to(self.args.device)

            for op_info in avai_ops:
                normalize_process_time = torch.tensor([op_info['process_time'] / max_process_time], dtype=torch.float32, device=self.args.device)
                score = torch.cat((score, torch.cat((x_dict['m'][op_info['m_id']], x_dict['op'][op_unfinished.index(op_info['node_id'])], normalize_process_time), dim=0).unsqueeze(0)), dim=0)

            for i in range(self.policy_num_layers - 1):
                score = F.leaky_relu(self.layers[i](score))
            score = self.layers[self.policy_num_layers - 1](score)

            probs = F.softmax(score, dim=0).flatten()
            dist = Categorical(probs)

            if greedy == True:
                idx = torch.argmax(score)
            else:
                idx = dist.sample()
            self.log_probs.append(dist.log_prob(idx))
            self.entropies.append(dist.entropy())
            return idx.item(), probs[idx].item()

        else:
             # Build mapping node_id -> index in op_unfinished to avoid .index calls
            nodeid_to_idx = {node_id: i for i, node_id in enumerate(op_unfinished)}

            # Preallocate lists for indices and process times
            m_ids = []
            op_idxs = []
            proc_times = []
            job_ids = []

            for d in avai_ops:
                m_ids.append(d['m_id'])
                op_idxs.append(nodeid_to_idx[d['node_id']])
                proc_times.append(d['process_time'])
                job_ids.append(d['job_id'])

            # convert to tensors on the correct device (do once)
            device = self.args.device
            m_ids_t = torch.tensor(m_ids, dtype=torch.long, device=device)       # (k,)
            op_idxs_t = torch.tensor(op_idxs, dtype=torch.long, device=device)   # (k,)
            proc_times_t = torch.tensor(proc_times, dtype=torch.float32, device=device) / max_process_time
            proc_times_t = proc_times_t.unsqueeze(1)  # (k,1)

            # gather features in batch (this avoids Python-loop concat)
            m_feats = x_dict['m'][m_ids_t]    # (k, feat_dim)
            op_feats = x_dict['op'][op_idxs_t]# (k, feat_dim)

            # concatenate once
            score = torch.cat((m_feats, op_feats, proc_times_t), dim=1)  # (k, feat_dim*2 + 1)

            # forward through policy layers (same as before but now batched)
            for i in range(self.policy_num_layers - 1):
                score = F.leaky_relu(self.layers[i](score))
            score = self.layers[self.policy_num_layers - 1](score)  # (k, 1?) or (k, D)

            probs = F.softmax(score.squeeze(-1), dim=0)  # (k,)
            max_select = data['m']['x'].size()[0]
            probs_cpu     = probs.cpu()                      # move to CPU if needed
            job_ids   = torch.tensor([d['job_id'] for d in avai_ops])
            mach_ids  = torch.tensor([d['m_id']  for d in avai_ops])
            
            if alpha == 0:
                valid_mask = probs_cpu > alpha
            else:
                valid_mask = probs_cpu > (1/alpha)
            
            if valid_mask.sum() <= 1:
                best_idx = torch.argmax(probs_cpu)
                best_action = avai_ops[best_idx.item()]
                return [best_action], probs_cpu[best_idx].item()


            # --- greedy “one-job–one-machine” selection -----------------
            actions          = []            # indices we finally keep
            selected = []
            NEG_INF = float('-inf')
            while True:
                # Check if any valid remains
                if not valid_mask.any():
                    break

                # mask out invalids by setting them to -inf
                # create masked view (no copy of full tensor needed aside from one assignment)
                # be careful: assign into a temp view to keep probs unchanged if needed
                masked_probs = probs_cpu.clone()                # small cost; can optimize to avoid clone if acceptable
                masked_probs[~valid_mask] = NEG_INF

                # find global maximum among masked entries
                idx = int(masked_probs.view(-1).argmax().item())
                max_val = float(masked_probs.view(-1)[idx].item())
                if max_val == NEG_INF:
                    break  # no valid candidate left

                # convert flat index to (job, machine)
                j = job_ids[idx].item()
                m = mach_ids[idx].item()

                # record selection as linear index or tuple
                selected.append(idx)
                actions.append(avai_ops[idx])

                # mask out chosen job row and machine column
                valid_mask[torch.where(job_ids==j)] = False
                valid_mask[torch.where(mach_ids==m)] = False

                # stop if we filled all possible assignments
                if len(selected) >= max_select:
                    break

            
            
            return actions, probs[idx].item()
            
            # probs_cpu     = probs.cpu()                      # move to CPU if needed
            # job_ids   = torch.tensor([d['job_id'] for d in avai_ops])
            # mach_ids  = torch.tensor([d['m_id']  for d in avai_ops])

            # # --- greedy “one-job–one-machine” selection -----------------
            # actions          = []            # indices we finally keep
            # used_jobs     = set()         # job_id already chosen
            # used_machines = set()         # m_id  already chosen

            # for idx in torch.argsort(probs_cpu, descending=True):  # highest-probability first
            #     j = job_ids[idx].item()
            #     m = mach_ids[idx].item()
            #     if j not in used_jobs and m not in used_machines:   # both still free?
            #         actions.append(avai_ops[idx.item()])
            #         used_jobs.add(j)
            #         used_machines.add(m)
            # return actions, probs[idx].item()

    
    def calculate_loss(self, device):
        loss = []
        returns = torch.FloatTensor(list(accumulate(self.rewards[::-1]))[::-1]).to(device)
        policy_loss = 0.0
        entropy_loss = 0.0

        for log_prob, entropy, R, baseline in zip(self.log_probs, self.entropies, returns, self.baselines):
            if baseline == 0:
                advantage = R * -1
            else:
                advantage = ((R - baseline) / baseline) * -1
            loss.append(-log_prob * advantage - self.args.entropy_coef * entropy)
            policy_loss += log_prob * advantage
            entropy_loss += entropy
        
        return torch.stack(loss).mean(), policy_loss / len(self.log_probs), entropy_loss / len(self.log_probs)
 
    def clear_memory(self):
        del self.log_probs[:]
        del self.entropies[:]
        del self.rewards[:]
        del self.baselines[:]
        return
         