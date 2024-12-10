import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp
from dictknife import deepmerge
from shared.should_save import should_save
from shared.tester import Tester

import wandb
from continuousp.data.crystal_dataset import CrystalDataset
from continuousp.data.dataset_partition import DatasetPartition
from continuousp.utils.logger import LOGGER


def test_on_one_gpu(
    device: torch.device,
    train_run_id: str,
    config: str,
    num_test_data: int,
    return_list: list,
    index: int,
    lock: mp.Lock,
) -> None:
    with lock:
        LOGGER.info(f'Testing on {device} for sample {index}')
        rms_dists = list(
            Tester(
                None,
                train_run_id,
                config,
                num_test_data,
                device,
            ).get_rms_dists(),
        )
        return_list[index] = rms_dists


def main(
    run_name: str,
    train_run_id: str,
    batch_size: int,
    num_samples: int,
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

    run = wandb.init(
        project='ContinuouSP-eval',
        name=run_name,
        config=config,
    )
    run.log_code(
        root=str(Path(__file__).parent.parent),
        include_fn=should_save,
    )

    mp.set_start_method('spawn', force=True)

    num_gpus = torch.cuda.device_count()

    assert num_gpus > 0

    manager = mp.Manager()
    return_list = manager.list([None] * num_samples)
    gpu_locks = [mp.Lock() for _ in range(num_gpus)]

    processes = []
    for sample_id in range(num_samples):
        device_id = sample_id % num_gpus
        p = mp.Process(
            target=test_on_one_gpu,
            args=(
                torch.device(f'cuda:{device_id}'),
                train_run_id,
                config,
                num_test_data,
                return_list,
                sample_id,
                gpu_locks[device_id],
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    if num_test_data == -1:
        num_test_data = len(CrystalDataset(config['training']['dataset'], DatasetPartition.TEST))

    match_counts = [0] * num_test_data
    min_rmse = [float('inf')] * num_test_data

    for rms_dists in return_list:
        for i, rms_dist in enumerate(rms_dists):
            if rms_dist is not None:
                match_counts[i] += 1
                min_rmse[i] = min(min_rmse[i], rms_dist)

    table_for_each_crystal = wandb.Table(columns=['Composition ID', 'Match Count', 'Min RMSE'])
    test_dataset = CrystalDataset(
        dataset=config['training']['dataset'],
        partition=DatasetPartition.TEST,
    )

    matched_data_count = 0
    sum_min_rmse = 0.0

    for i in range(num_test_data):
        if match_counts[i] > 0:
            table_for_each_crystal.add_data(
                test_dataset[i].composition_id,
                match_counts[i],
                min_rmse[i],
            )
            matched_data_count += 1
            sum_min_rmse += min_rmse[i]
        else:
            table_for_each_crystal.add_data(
                test_dataset[i].composition_id,
                0,
                -1,
            )
    run.log({'For each crystal': table_for_each_crystal})

    table_whole = wandb.Table(columns=['Match Rate', 'Average Min RMSE over Matched Data'])
    if matched_data_count > 0:
        match_rate = matched_data_count / num_test_data * 100
        avg_min_rmse = sum_min_rmse / matched_data_count
        table_whole.add_data(match_rate, avg_min_rmse)
    else:
        table_whole.add_data(0, -1)
    run.log({'Whole': table_whole})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('train_run_id')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('num_samples', type=int)
    parser.add_argument('num_test_data', type=int, default=-1)
    args = parser.parse_args()
    main(
        args.run_name,
        args.train_run_id,
        args.batch_size,
        args.num_samples,
        args.num_test_data,
    )
