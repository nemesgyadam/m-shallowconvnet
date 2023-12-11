import torch

from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from utils.training_utils import (
    get_criterion,
    get_scheduler,
    get_optimizer
)
from model.mshallowconvnet import get_model

from tqdm.notebook import tqdm


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
        self.val_accs = []

    
  
 
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.sample_batch = batch['data'].type(torch.float)


        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']

        if 'subject_info' in batch.keys():
            eeg_input = batch['data'].type(torch.float)
            subject_info = batch['subject_info'].type(torch.long)
            outputs = self(eeg_input, subject_info)
        else:
            eeg_input = batch['data'].type(torch.float)
            outputs = self(eeg_input)


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

    # def on_train_epoch_end(self):
    #     if self.current_epoch == 0:
    #         self.logger.experiment.add_graph(self.model, self.sample_batch)



    def on_fit_start(self):
        self.pbar = tqdm(total=self.args['EPOCHS'])

    def on_train_epoch_end(self):
        self.pbar.update(1)
    
    def evaluate(self, batch, stage=None):
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']

        if 'subject_info' in batch.keys():
            eeg_input = batch['data'].type(torch.float)
            subject_info = batch['subject_info'].type(torch.long)
            outputs = self(eeg_input, subject_info)
        else:
            eeg_input = batch['data'].type(torch.float)
            outputs = self(eeg_input)
        
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1), num_classes=4, task = 'multiclass')
        self.val_accs.append(acc)
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
        if 'subject_info' in batch.keys():
            eeg_input = batch['data'].type(torch.float)
            subject_info = batch['subject_info'].type(torch.long)
            return self(eeg_input, subject_info)
        else:
            eeg_input = batch['data'].type(torch.float)
            return self(eeg_input)
    
    
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

