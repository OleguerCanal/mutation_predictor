
import torch
from torch import nn
import pytorch_lightning as pl

from .transformer_ingredients import TransformerEncoder

class MutationPredictor(pl.LightningModule):
    def __init__(self,
                input_dim=18,
                output_dim=4,
                d_model=256,
                d_ff=512,
                num_layers=4,
                num_heads=4,
                dropout_p=0.3,
                lr=1.5e-4):
        self.lr = lr
        self.transformer_encoder = TransformerEncoder(input_dim=input_dim,
                                                      d_model=d_model,
                                                      d_ff=d_ff,
                                                      num_layers=num_layers,
                                                      num_heads=num_heads,
                                                      dropout_p=dropout_p)
        self.output_layer = nn.Linear(d_model, out_features)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_batch):
        x, _ = self.transformer_encoder(input_batch)
        output = self.output_layer(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(batch, batch_indx):
        inputs, labels = batch
        predictions = self(inputs)
        loss = self.loss(predictions, labels)
        return loss

    def training_step(self, batch, batch_indx):
        return self._step(batch, batch_indx)

    def validation_step(self, batch, batch_indx):
        return self._step(batch, batch_indx)

    def test_step(self, batch, batch_indx):
        return self._step(batch, batch_indx)
