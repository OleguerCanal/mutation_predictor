
import torch
from torch import nn
import pytorch_lightning as pl
import wandb

from .transformer_ingredients import TransformerEncoder

class MutationPredictor(pl.LightningModule):
    def __init__(self,
                input_dim=18,
                seq_len=41,
                output_dim=4,
                d_model=256,
                d_ff=512,
                num_layers=4,
                num_heads=4,
                dropout_p=0.3,
                lr=1.0e-2):
        super().__init__()
        self.lr = lr
        self.transformer_encoder = TransformerEncoder(input_dim=input_dim,
                                                      d_model=d_model,
                                                      d_ff=d_ff,
                                                      num_layers=num_layers,
                                                      num_heads=num_heads,
                                                      dropout_p=dropout_p)
        self.output_layer = nn.Linear(d_model*seq_len, output_dim)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.val_preds = []
        self.val_labels = []

    def forward(self, input_batch):
        batch_size = input_batch.shape[0]
        x, _ = self.transformer_encoder(input_batch)
        x = x.view(batch_size, -1)
        output = self.softmax(self.output_layer(x))
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_indx):
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.loss(predictions, labels)
        wandb.log({"loss/train": loss})
        return loss

    def validation_step(self, batch, batch_indx):
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.loss(predictions, labels)
        # wandb.log({"loss/val": loss})
        self.log("loss/val", loss)

        pred_indexes = torch.argmax(predictions, dim=-1).detach().cpu().numpy().tolist()
        labels_indexes = labels.detach().cpu().numpy().tolist()

        self.val_preds += pred_indexes
        self.val_labels += labels_indexes
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        alphabet = ["A", "C", "G", "T"]
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=self.val_labels, preds=self.val_preds,
                        class_names=alphabet)})
        self.val_preds = []
        self.val_labels = []


    def test_step(self, batch, batch_indx):
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.loss(predictions, labels.squeeze(1))
        # wandb.log({"loss/test": loss})
        self.log("loss/test", loss)
        return loss
