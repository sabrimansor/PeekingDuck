
import hydra
from omegaconf import DictConfig, OmegaConf

from peekingduck.training.model import Model
from peekingduck.training.trainer import Trainer


class DataLoader:
    def __init__(self):
        self.validation_set = None
        self.training_set = None

    def train_dataloader(self):
        return self.training_set

    def valid_dataloader(self):
        return self.validation_set


def run(cfg: DictConfig) -> None:

    # TODO Mock up DataLoader Object
    dl = DataLoader()
    train_loader = dl.train_dataloader()
    valid_loader = dl.valid_dataloader()

    # TODO Support Early Stopping and Checkpoint
    callbacks = cfg.callback_params.callbacks

    model = Model(cfg.model)
    trainer = Trainer(cfg.trainer, model=model, callbacks=callbacks)
    history = trainer.fit(train_loader, valid_loader)

@hydra.main(config_path='peekingduck/training', config_name='pipeline_config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()