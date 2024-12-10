from pathlib import Path

import torch
from shared.should_save import should_save
from torch.optim import Adam
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

import wandb
from continuousp.data.collate_crystals import collate_crystals
from continuousp.data.crystal_dataset import CrystalDataset
from continuousp.data.dataset_partition import DatasetPartition
from continuousp.models.continuous_energy_predictor import ContinuousEnergyPredictor
from continuousp.models.crystals import Crystals
from continuousp.utils.logger import LOGGER


class Trainer:
    def __init__(
        self,
        run_name: str | None,
        config: dict,
        num_train_data: int,
        num_valid_data: int,
        pretrain_run_id: str | None,
        device: torch.device,
    ) -> None:
        self.model = ContinuousEnergyPredictor(**config['model']).to(device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
        )

        train_dataset = CrystalDataset(
            dataset=config['training']['dataset'],
            partition=DatasetPartition.TRAIN,
        )
        valid_dataset = CrystalDataset(
            dataset=config['training']['dataset'],
            partition=DatasetPartition.VALID,
        )
        if num_train_data != -1:
            train_dataset = Subset(
                train_dataset,
                torch.randperm(len(train_dataset))[:num_train_data],
            )
        if num_valid_data != -1:
            valid_dataset = Subset(
                valid_dataset,
                torch.randperm(len(valid_dataset))[:num_valid_data],
            )

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_crystals,
        )
        self.valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_crystals,
        )

        self.run = wandb.init(
            project='ContinuouSP-train',
            name=run_name,
            config=config,
            mode='disabled' if run_name is None else 'online',
        )
        self.run.log_code(
            root=str(Path(__file__).parent.parent.parent),
            include_fn=should_save,
        )
        self.run.watch(self.model, log='all', log_freq=1)

        self.num_epochs = config['training']['num_epochs']
        self.regularization_lambda = config['training']['regularization_lambda']
        self.snapshot_path = (Path('./snapshots') / self.run.id).with_suffix('.pth')

        if pretrain_run_id is not None:
            snapshot_path = (Path('./snapshots') / pretrain_run_id).with_suffix('.pth')
            self._load_snapshot(snapshot_path)

        self.device = device

    def __del__(self) -> None:
        self.run.finish()

    def _load_snapshot(self, snapshot_path: str) -> None:
        LOGGER.info('Loading snapshot')
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])

    def _save_snapshot(self, epoch: int) -> None:
        snapshot = {
            'MODEL_STATE': self.model.state_dict(),
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            'EPOCH': epoch,
        }
        Path(self.snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, self.snapshot_path)

    def _run_batch(self, source: Crystals, *, train: bool) -> float:
        source = source.to(self.device)
        self.model.eval()
        atomic_numbers, natoms = source.geometrically_multiplied_composition()
        sample = list(
            self.model.generate(atomic_numbers, natoms, source.composition_id),
        )[-1]
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            energy_real = self.model.energy_with_penalty(source)
            energy_sample = self.model.energy_with_penalty(sample)
            loss_cd = (energy_real - energy_sample).mean()
            loss_rg = (energy_real**2).mean() + (energy_sample**2).mean()
            loss = loss_cd + self.regularization_lambda * loss_rg
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        self.run.define_metric('Batch')
        self.run.define_metric('Epoch')
        self.run.define_metric(
            'Train loss / Batch',
            step_metric='Batch',
        )
        self.run.define_metric(
            'Train loss / Epoch',
            step_metric='Epoch',
        )
        self.run.define_metric(
            'Test loss / Epoch',
            step_metric='Epoch',
        )
        batch = 0
        for epoch in range(self.num_epochs):
            total_train_loss = 0.0
            total_valid_loss = 0.0
            for source in self.train_data_loader:
                LOGGER.debug(f'Epoch: {epoch}, Batch: {batch}')
                train_loss = self._run_batch(source, train=True)
                self.run.log({'Train loss / Batch': train_loss, 'Batch': batch})
                total_train_loss += train_loss
                batch += 1
            self._save_snapshot(epoch)
            for source in self.valid_data_loader:
                total_valid_loss += self._run_batch(source, train=False)
            self.run.log(
                {
                    'Train loss / Epoch': total_train_loss / len(self.train_data_loader),
                    'Valid loss / Epoch': total_valid_loss / len(self.valid_data_loader),
                    'Epoch': epoch,
                },
            )
