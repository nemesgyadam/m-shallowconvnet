import torch

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from utils.training_utils import (
    get_criterion,
    get_scheduler,
    get_optimizer
)
from model.mshallowconvnet import get_model


class LitModel(LightningModule):
    def __init__(self, args, model = None):
        super().__init__()
        
        self.save_hyperparameters()
        if model:
            self.model = model
        else:   
            self.model = get_model(args)
        self.criterion = get_criterion()
        self.args = args
    
    
    def forward(self, eeg_data, subject_id):
        return self.model(eeg_data, subject_id)
    
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.sample_batch = batch['data'].type(torch.float)

        eeg_input = batch['data'].type(torch.float)
        subject_id = batch['subject_id'].type(torch.long)
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']
        
        outputs = self(eeg_input, subject_id)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1), num_classes=4, task = 'multiclass')
        self.log('train_loss', loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True)
        self.log('train_acc', acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True)
        return {'loss': loss}

    
    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(self.model, self.sample_batch)
    
    
    def evaluate(self, batch, stage=None):
        eeg_input = batch['data'].type(torch.float)
        subject_id = batch['subject_id'].type(torch.long)
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']
        
        outputs = self(eeg_input, subject_id)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1), num_classes=4, task = 'multiclass')
        
        if stage:
            self.log(f'{stage}_loss', loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
            self.log(f'{stage}_acc', acc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True)
    
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')
    
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        eeg_input = batch['data'].type(torch.float)
        subject_id = batch['subject_id'].type(torch.long)
        return self(eeg_input, subject_id)
    
    
    def configure_optimizers(self):
        self.__init_optimizers()
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler_config}
    
    
    def __init_optimizers(self):
        self.optimizer = get_optimizer(self, self.args)
        self.lr_scheduler = get_scheduler(self.optimizer, self.args)
        self.lr_scheduler_config = {
            'scheduler': self.lr_scheduler,
            'internal': 'epoch'
        }


def get_litmodel(args, model=None):
    
    model = LitModel(args, model = model)
    
    return model

