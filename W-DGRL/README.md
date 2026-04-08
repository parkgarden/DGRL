# W-DGRL

Wide-graph variant of DGRL from the paper:
> **"Deep Reinforcement Learning with Dynamic Graph Pruning for Scalable Flexible Job Shop Scheduling"**

Uses a Dual Attention Network (DAN) with operation and machine message attention blocks, trained with PPO. Dynamic graph pruning limits the active subgraph to a `graph_len`-operation window per job. At inference, Multi-Action (MA) selection dispatches multiple non-conflicting (job, machine) pairs per step via greedy bipartite matching.

---

## Requirements

```bash
conda env create -f environment_local.yml
```

Or install manually:
```bash
pip install torch torch-scatter tqdm numpy pandas
```

---

## Training

```bash
python train_DGRL.py --n_j 10 --n_m 5 --data_source SD2 --dyn 1 --graph_len 5
```

| Argument | Description | Default |
|---|---|---|
| `--n_j` | Number of jobs | 10 |
| `--n_m` | Number of machines | 5 |
| `--data_source` | `SD1` or `SD2` data type | `SD2` |
| `--dyn` | Dynamic graph pruning: `1`=enabled, `0`=disabled | 0 |
| `--graph_len` | Graph window size (ops per job visible) | 5 |

Training data is generated on-the-fly from `./data/data_train_vali/<data_source>/`. Best checkpoint saved to `./trained_network/<data_source>/<model_name>.pth`.

---

## Testing

```bash
python test_DGRL_W.py --data_source 10020 --dyn 1 --graph_len 5 --action_threshold 5 --test_model 10x5+mix+dyn1
```

| Argument | Description | Default |
|---|---|---|
| `--data_source` | Test dataset subfolder under `./data/` | `10020` |
| `--model_source` | Subfolder under `./trained_network/` | `SD2` |
| `--test_model` | Model filename(s) without `.pth` | `10x5+mix` |
| `--dyn` | Dynamic graph pruning: `1`=enabled, `0`=disabled (must match training) | 0 |
| `--graph_len` | Must match training | 5 |
| `--action_threshold` | MA threshold denominator (prob > `1/K`); `0`=single-action | 0 |
| `--ins_start/end` | Index range of test instances | 0 / 10 |
| `--flag_sample` | `1` for sampling mode (DRL-S), `0` for greedy (DRL-G) | 0 |

Results saved as `.npy` files under `./test_results/<data_source>/`.

---

## Data

Place FJSP instances in:
- Training/validation: `./data/data_train_vali/<data_source>/<n_j>x<n_m>/`
- Testing: `./data/<data_source>/<n_j>x<n_m>/` or directly under `./data/<data_source>/`
