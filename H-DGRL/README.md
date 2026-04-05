# H-DGRL

Heterogeneous-graph variant of DGRL from the paper:
> **"Deep Reinforcement Learning with Dynamic Graph Pruning for Scalable Flexible Job Shop Scheduling"**

Uses a Heterogeneous GNN (HeteroConv with GINEConv layers over op→op, op→m, m→op, m→m edges) trained with REINFORCE. Dynamic graph pruning limits the active subgraph to a `graph_len`-operation window per job. At inference, Multi-Action (MA) mode selects multiple non-conflicting (job, machine) pairs per step via greedy bipartite matching.

---

## Requirements

```bash
conda env create -f environment.yml
```

Or install manually:
```bash
pip install torch torch-geometric tensorboard pandas openpyxl
```

---

## Training

```bash
python train.py --dyn 1 --graph_len 5 --date train --instance_type FJSP --data_size 10 --delete_node true
```

| Argument | Description | Default |
|---|---|---|
| `--dyn` | Dynamic graph env (`1`) or static (`0`) | 0 |
| `--graph_len` | Graph window size | 10 |
| `--instance_type` | `FJSP` or `JSP` | FJSP |
| `--data_size` | Number of jobs in training instances | 10 |
| `--episode` | Number of training episodes | 300001 |
| `--lr` | Learning rate | 1e-4 |

Training data is generated on-the-fly. Validation runs every 1000 episodes against `./datasets/FJSP/data_dev/1510/`. Best checkpoint is saved to `./weight/dyn{dyn}_glen{graph_len}/best`.

---

## Testing

```bash
python test_DGRL.py --data_source Brandimarte_Data --dyn 1 --graph_len 5 --mode 1 --alpha 5
```

| Argument | Description | Default |
|---|---|---|
| `--data_source` | Subfolder under `./datasets/FJSP/` | `Brandimarte_Data` |
| `--dyn` | Must match training | 0 |
| `--graph_len` | Must match training | 10 |
| `--mode` | `0`: single-action, `1`: MA with DG, `2`: MA w/o DG, `3`: single-action w/o MA | 0 |
| `--alpha` | MA threshold denominator (prob > `1/alpha`); `0` disables MA | 0 |
| `--n_ins` | Number of instances to test; `0` for all | 10 |

Results are saved as `.txt` logs and an Excel file under `./result/<data_source>/`.

---

## Data

Place FJSP benchmark instances in `./datasets/FJSP/<data_source>/`. Validation instances go in `./datasets/FJSP/data_dev/1510/`.
