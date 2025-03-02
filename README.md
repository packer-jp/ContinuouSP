# ContinuouSP
## Setup
We adopt [Rye](https://rye-up.com/) for project/package management. Please follow the instructions in the link to install Rye, and then run the following command to create a virtual environment under `.venv`:

```bash
rye sync
```

In order to apply formatters/linters at commit time, please run the following command once after creating the virtual environment:

```bash
rye run pre-commit install
```

We utilize [Weights & Biases](https://wandb.ai/site) for experiment tracking. Please sign up for an account.

## Execution
General form:
```bash
rye run python exec/train.py <Run name on W&B> <Config name> [<Size of train dataset>] [<Size of valid dataset>] [<Run ID on W&B of pretrained model>]
rye run python exec/test.py <Run name on W&B> <Run ID on W&B of trained model> <Batch size> [<Size of test dataset>]
rye run python exec/eval.py <Run name on W&B> <Run ID on W&B of trained model> <Batch size> [<Number of samples per test datum>] [<Size of Test dataset>]
```

Examples:
```bash
rye run python exec/train.py MP-20 mp_20
rye run python exec/test.py MP-20 8grneu3c 128
rye run python exec/eval.py MP-20 8grneu3c 128 20
```

## Trained Models
| Dataset | Path |
| - | - |
| Perov-5 | `snapshots/0f1rvcsu.pth` |
| MP-20 | `snapshots/8grneu3c.pth` |
| MPTS-52 | `snapshots/b8skv8g3.pth` |
