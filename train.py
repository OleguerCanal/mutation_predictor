import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data_modules.data_module import MutationsDatamodule
from model.transformer_ingredients import set_seed
from model.mutation_predictor import MutationPredictor

set_seed(0)

if __name__=="__main__":
    datamodule = MutationsDatamodule(dataset_path="data/data.csv", batch_size=10)
    wandb_logger = WandbLogger(project="mutation_predictor")

    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", logger=wandb_logger)
    model = MutationPredictor()

    trainer.fit(model, datamodule)