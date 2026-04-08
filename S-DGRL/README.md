# S-DGRL

Standard variant of DGRL from the paper:
> **"Deep Reinforcement Learning with Dynamic Graph Pruning for Scalable Flexible Job Shop Scheduling"**

Uses a Heterogeneous GNN (HGNN) with PPO for FJSP. Dynamic graph pruning limits the active subgraph to a `graph_len`-operation window per job for scalability. At inference, Multi-Action (MA) selection dispatches multiple non-conflicting (job, machine) pairs per step via greedy bipartite matching.

---

## Requirements

```bash
conda env create -f environment.yml
```

---

## Training

```bash
python train.py --n_j 10 --n_m 5 --dyn 1 --graph_len 10
```

| Argument | Description | Default |
|---|---|---|
| `--n_j` | Number of jobs | 10 |
| `--n_m` | Number of machines | 5 |
| `--dyn` | Dynamic graph env (`1`) or static (`0`) | 0 |
| `--graph_len` | Graph window size | 10 |

Best checkpoint is saved under `./save/train_<timestamp>/`. Copy it to `./model/DGRL/` before testing.

---

## Testing

```bash
python test_DGRL_S.py --data_source 10020 --ins_start 0 --ins_end 10 --ma 1 --alpha 5
```

| Argument | Description | Default |
|---|---|---|
| `--data_source` | Subfolder under `./data_test/` | `10020` |
| `--ins_start/end` | Index range of test instances | 0 / 10 |
| `--ma` | Multi-action selection (`1`) or single-action (`0`) | 1 |
| `--alpha` | MA threshold denominator (pairs with prob > `1/alpha` selected); `0` disables MA | 0 |
| `--graph_len` | Must match training | 10 |
| `--num_average` | Repeated runs per instance (DRL-G) | 1 |

Toggle `sample` in `config.json` to switch between DRL-G (greedy) and DRL-S (sampling) modes. Results are saved as Excel under `./save/test_<dataset>_DGRL_<config>_<timestamp>/`.

---

## Data

Place FJSP benchmark instances in `./data_test/<data_source>/`. To generate random instances:

```bash
python create_ins.py
```
