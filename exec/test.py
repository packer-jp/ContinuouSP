import argparse

import torch
from dictknife import deepmerge
from shared.tester import Tester

import wandb


def main(
    run_name: str,
    train_run_id: str,
    batch_size: int,
    num_test_data: int,
) -> None:
    try:
        train_config = wandb.Api().run(f'ContinuouSP-train/{train_run_id}').config
    except wandb.errors.CommError:
        train_config = wandb.Api().run(f'ContinuouSP-energy-pred-train/{train_run_id}').config

    config = deepmerge(
        train_config,
        {
            'testing': {
                'train_run_id': train_run_id,
                'batch_size': batch_size,
                'num_test_data': num_test_data,
            },
        },
    )
    tester = Tester(
        run_name,
        train_run_id,
        config,
        num_test_data,
        'cuda' if torch.cuda.is_available() else 'cpu',
        shuffle=True,
    )

    tester.distance_vs_energy()
    tester.magnification_vs_energy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('train_run_id')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('num_test_data', type=int, nargs='?', default=-1)
    args = parser.parse_args()
    main(
        args.run_name,
        args.train_run_id,
        args.batch_size,
        args.num_test_data,
    )
