# DGRL

Official implementation of the paper:
> **”Deep Reinforcement Learning with Dynamic Graph Pruning for Scalable Flexible Job Shop Scheduling”**

This repository contains three variants of the DGRL framework for the Flexible Job Shop Scheduling Problem (FJSP), each exploring a different graph neural network architecture combined with dynamic graph pruning to improve scalability:

| Variant | Architecture | Training |
|---|---|---|
| [S-DGRL](./S-DGRL/) | Heterogeneous GNN (HGNN) | PPO |
| [H-DGRL](./H-DGRL/) | Heterogeneous GNN (HeteroConv + GINEConv) | REINFORCE |
| [W-DGRL](./W-DGRL/) | Dual Attention Network (DAN) | PPO |

Dynamic graph pruning limits the active operation subgraph to a sliding window of `graph_len` operations per job, enabling the models to scale to large instances without growing the graph size proportionally. At inference, a novel Multi-Action (MA) selection mechanism dispatches multiple non-conflicting (job, machine) pairs per step via greedy bipartite matching, significantly reducing the number of decision steps and computation cost without compromising solution quality.

---

## Data

Test datasets are provided in the `./data/` directory as zip archives. Extract them into the appropriate subfolder expected by each variant before testing:

| Archive | Description |
|---|---|
| `10020.zip` – `200100.zip` | Random FJSP instances (jobs × machines) |
| `Brandimarte_Data.zip` | Brandimarte benchmark instances |
| `vdata.zip` | Validation data |

Refer to the README of each variant for the exact data path expected.
