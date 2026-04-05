import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch.distributions import Categorical
from itertools import accumulate
from model.gnn import GNN
torch.set_printoptions(precision=10)

class REINFORCE_DM(nn.Module):
    def __init__(self, args):
        super(REINFORCE_DM, self).__init__()
        self.args = args
        self.policy_num_layers = args.policy_num_layers
        self.hidden_dim = args.hidden_dim
        self.gnn = GNN(args)
        self.layers = torch.nn.ModuleList()  # policy network
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

    def forward(self, avai_ops, data, op_unfinished, max_process_time, greedy=False, mode=0, alpha=0):
        x_dict = self.gnn(data)

        # Build mapping node_id -> index in op_unfinished
        nodeid_to_idx = {node_id: i for i, node_id in enumerate(op_unfinished)}

        m_ids, op_idxs, proc_times, job_ids = [], [], [], []
        for d in avai_ops:
            m_ids.append(d['m_id'])
            op_idxs.append(nodeid_to_idx[d['node_id']])
            proc_times.append(d['process_time'])
            job_ids.append(d['job_id'])

        device = self.args.device
        m_ids_t = torch.tensor(m_ids, dtype=torch.long, device=device)
        op_idxs_t = torch.tensor(op_idxs, dtype=torch.long, device=device)
        proc_times_t = torch.tensor(proc_times, dtype=torch.float32, device=device) / max_process_time
        proc_times_t = proc_times_t.unsqueeze(1)

        m_feats = x_dict['m'][m_ids_t]
        op_feats = x_dict['op'][op_idxs_t]
        score = torch.cat((m_feats, op_feats, proc_times_t), dim=1)

        for i in range(self.policy_num_layers - 1):
            score = F.leaky_relu(self.layers[i](score))
        score = self.layers[self.policy_num_layers - 1](score)

        probs = F.softmax(score.squeeze(-1), dim=0)

        # Single-action mode
        if mode == 0:
            dist = Categorical(probs)
            idx = torch.argmax(score) if greedy else dist.sample()
            self.log_probs.append(dist.log_prob(idx))
            self.entropies.append(dist.entropy())
            return idx.item(), probs[idx].item()

        # Multi-action mode: greedy bipartite matching (one job, one machine per selection)
        else:
            max_select = data['m']['x'].size()[0]
            probs_cpu = probs.cpu()
            job_ids = torch.tensor([d['job_id'] for d in avai_ops])
            mach_ids = torch.tensor([d['m_id'] for d in avai_ops])

            valid_mask = probs_cpu > (1 / alpha) if alpha != 0 else probs_cpu > alpha

            if valid_mask.sum() <= 1:
                best_idx = torch.argmax(probs_cpu)
                return [avai_ops[best_idx.item()]], probs_cpu[best_idx].item()

            actions, selected = [], []
            NEG_INF = float('-inf')
            while True:
                if not valid_mask.any():
                    break
                masked_probs = probs_cpu.clone()
                masked_probs[~valid_mask] = NEG_INF
                idx = int(masked_probs.view(-1).argmax().item())
                if masked_probs.view(-1)[idx].item() == NEG_INF:
                    break
                j = job_ids[idx].item()
                m = mach_ids[idx].item()
                selected.append(idx)
                actions.append(avai_ops[idx])
                # Enforce bipartite constraint: remove chosen job and machine
                valid_mask[torch.where(job_ids == j)] = False
                valid_mask[torch.where(mach_ids == m)] = False
                if len(selected) >= max_select:
                    break

            return actions, probs[idx].item()

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
