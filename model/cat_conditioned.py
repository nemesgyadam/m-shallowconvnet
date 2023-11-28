import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from model.mshallowconvnet import MShallowConvNet
from model.conditioned_batch_norm import ConditionedBatchNorm
from model.subject_encoder import SubjectEncoder



class CatConditioned(nn.Module):
    def __init__(
        self,
        args: dict,
        n_subjects: int = 9,
        # SubjectEncoder params
        subject_filters: int = 16,
        # Dense params
        final_features: int = 4,
        dropout_rate: float = 0.2,
        n_classes: int = 4,
        device: str = "cpu",
    ) -> None:

        super(CatConditioned, self).__init__()
        self.device = device


        ''' EEG Encoder '''
        self.eeg_encoder = MShallowConvNet(args['num_channels'], args['sampling_rate'])
        self.eeg_dim = self.eeg_encoder.calculate_output_dim()
        self.eeg_bn = ConditionedBatchNorm(self.eeg_dim, n_subjects)

        ''' Subject Encoder '''
        self.subject_encoder = nn.Embedding(n_subjects, subject_filters)
        self.subject_bn = ConditionedBatchNorm(subject_filters, n_subjects)

        ''' Conditioning '''
        self.linear = nn.Linear(subject_filters+self.eeg_dim, final_features)
        self.act = nn.ELU()

        ''' Initialize Classifier '''
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_features, n_classes)

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_id:torch.Tensor) -> torch.Tensor:
        #eeg_data, subject_id = x
        ''' Encoders '''
    
        eeg_features = self.eeg_encoder(eeg_data.float())
        #eeg_features = self.eeg_bn(eeg_features, subject_id)   #TODO put it back

        subject_features = self.subject_encoder(subject_id.int())
        #subject_features = self.subject_bn(subject_features, subject_id) #TODO put it back

        ''' Conditioning '''
        x = torch.cat((eeg_features, subject_features), dim=1)       
        x = self.linear(x)
        x = self.act(x)

        ''' Classify '''
        x = self.dropout(x)
        x = self.classifier(x)
        return x



    @classmethod
    def from_config(cls, config, device="cpu"):
        # Get the signature of the class constructor
        signature = inspect.signature(cls.__init__)

        # Remove the 'self' parameter
        parameters = signature.parameters.copy()
        del parameters["self"]

        # Extract the necessary arguments from the config
        kwargs = {name: config[name] for name in parameters.keys() if name in config}
        kwargs["device"] = device
        
        return cls(**kwargs)
