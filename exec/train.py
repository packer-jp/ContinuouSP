import argparse
import importlib

import torch
from dictknife import deepmerge
from shared.trainer import Trainer

from continuousp.utils.logger import LOGGER


def main(
    run_name: str,
    config_name: str,
    num_train_data: int,
    num_valid_data: int,
    pretrain_run_id: str | None,
) -> None:
    config = deepmerge(
        importlib.import_module('configs.common').config,
        importlib.import_module(f'configs.{config_name}').config,
        {
            'training': {
                'num_train_data': num_train_data,
                'num_valid_data': num_valid_data,
            },
        },
    )
    Trainer(
        run_name,
        config,
        num_train_data,
        num_valid_data,
        pretrain_run_id,
        'cuda' if torch.cuda.is_available() else 'cpu',
    ).train()


if __name__ == '__main__':
    LOGGER.debug('Starting training')
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('config_name')
    parser.add_argument('num_train_data', type=int)
    parser.add_argument('num_valid_data', type=int)
    parser.add_argument('pretrain_run_id', type=str, nargs='?', default=None)
    args = parser.parse_args()

    main(
        args.run_name,
        args.config_name,
        args.num_train_data,
        args.num_valid_data,
        args.pretrain_run_id,
    )
