import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from model.mshallowconvnet import MShallowConvNet
from model.conditioned_batch_norm import ConditionedBatchNorm
from model.subject_encoder import SubjectEncoder



class ATTNConditionedSubjFtr(nn.Module):
    def __init__(
        self,
        args: dict,
        n_subjects: int = 9,
        # SubjectEncoder params

        subject_dim: int = 12, 
        embed_dim: int = 16,    # AKA subject_filters
        # Dense params
        #final_features: int = 4,
        dropout_rate: float = 0.2,
        n_classes: int = 4,
        device: str = "cpu",
    ) -> None:

        super(ATTNConditionedSubjFtr, self).__init__()
        self.device = device
        self.embed_dim = embed_dim


        ''' EEG Encoder '''
        self.eeg_encoder = MShallowConvNet(args['num_channels'], args['sampling_rate'])
        self.eeg_dim = self.eeg_encoder.calculate_output_dim()
        self.eeg_bn = ConditionedBatchNorm(self.eeg_dim, n_subjects)
        self.eeg_dim_reduction = nn.Linear(self.eeg_dim, self.embed_dim)

        ''' Subject Encoder '''
        self.subject_dim = subject_dim
        self.subject_encoder = nn.Linear(
            18, self.subject_dim
        )  # 18 is the number of features 
        self.subject_bn = ConditionedBatchNorm(self.subject_dim, n_subjects)
        self.subject_norm = nn.LayerNorm(self.subject_dim)

        self.subject_dim_reduction = nn.Linear(self.subject_dim, self.embed_dim)
        self.act1 = nn.ELU()

        ''' Conditioning '''
        #self.linear = nn.Linear(subject_filters+self.eeg_dim, final_features)
        self.act = nn.ELU()

        ''' Initialize Classifier '''
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.embed_dim, n_classes)

        #TODO weight init 

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_info:torch.Tensor) -> torch.Tensor:

        ''' Encoders '''
        eeg_features = self.eeg_encoder(eeg_data.float())
        #eeg_features = self.eeg_bn(eeg_features, subject_id)   #TODO put it back
        eeg_features = self.eeg_dim_reduction(eeg_features)


        subject_features = self.subject_encoder(subject_info.float())
        # subject_features = self.subject_norm(subject_features)
        # subject_features = self.subject_bn(subject_features, subject_numbers) # SUBJECT ID required for BN
        subject_features = self.subject_dim_reduction(subject_features)

        ''' Conditioning '''
        subject_features = nn.functional.sigmoid(subject_features)
        x = torch.mul(subject_features, eeg_features)
        x = self.act(x)


        ''' Classify '''
        x = self.dropout(x)
        x = self.classifier(x)
        return x
