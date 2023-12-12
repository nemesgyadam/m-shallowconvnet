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
        config: dict,
        num_subjects: int = 9,
        # EEG Encoder params
        eeg_normalization: str = 'None',  # Options: 'None', 'CondBatchNorm', 'LayerNorm'
        eeg_activation: bool= True,
        # Subject Encoder params
        subj_dim: int = 16,
        embedding_dimension: int = 16,
        subject_normalization: str = 'None',  # Options: 'None', 'CondBatchNorm', 'LayerNorm'
        # Classifier params
        dropout_probability: float = 0.2,
        combined_features_dimension: Optional[int] = None,
        num_classes: int = 4,
        device: str = "cpu",
    ) -> None:

        super().__init__()
        self.device = device
        self.subj_dim =subj_dim
        self.subj_dim = subj_dim
        self.embedding_dimension = embedding_dimension
        self.combined_features_dimension = combined_features_dimension
        self.eeg_normalization = eeg_normalization
        self.eeg_activation = eeg_activation
        self.subject_normalization = subject_normalization
     


        ''' EEG Encoder '''
        self.eeg_encoder = MShallowConvNet(config['num_channels'], config['sampling_rate'])
        if self.eeg_activation:
            self.eeg_act = nn.ELU()
        self.eeg_output_dimension = self.eeg_encoder.calculate_output_dim()
        if eeg_normalization == 'CondBatchNorm':
            self.eeg_normalization_layer = ConditionedBatchNorm(self.eeg_output_dimension, num_subjects)
        elif eeg_normalization == 'LayerNorm':
            self.eeg_normalization_layer = nn.LayerNorm(self.eeg_output_dimension)
        self.eeg_dimension_reduction = nn.Linear(self.eeg_output_dimension, self.embedding_dimension)

       

        ''' Subject Encoder '''
        self.subj_dim = subj_dim
        self.subject_encoder = nn.Linear(
            18, self.subj_dim
        )  # 18 is the number of features 
        if subject_normalization == 'CondBatchNorm':
            self.subject_normalization_layer = ConditionedBatchNorm(self.subj_dim, num_subjects)
        elif subject_normalization == 'LayerNorm':
            self.subject_normalization_layer = nn.LayerNorm(self.subj_dim)

        self.subj_dim_reduction = nn.Linear(self.subj_dim, self.embedding_dimension)
        self.act1 = nn.ELU()

        ''' Conditioning '''
        self.cond_act = nn.ELU()

        ''' Classifier '''
        if combined_features_dimension:
            self.final_linear = nn.Linear(self.embedding_dimension,combined_features_dimension)
            self.final_act = nn.ELU()
            self.dropout = nn.Dropout(dropout_probability)
            self.classifier = nn.Linear(combined_features_dimension, num_classes)
        else:
            self.dropout = nn.Dropout(dropout_probability)
            self.classifier = nn.Linear(self.embedding_dimension, num_classes)

        self.to(self.device)

    def forward(self, eeg_data: torch.Tensor, subject_info:torch.Tensor) -> torch.Tensor:

        ''' Encoders '''
        eeg_features = self.eeg_encoder(eeg_data.float())
        if self.eeg_activation:
            eeg_features = self.eeg_act(eeg_features)

        if self.eeg_normalization == 'CondBatchNorm':
            eeg_features = self.eeg_normalization_layer(eeg_features, subject_id)
        elif self.eeg_normalization == 'LayerNorm':
            eeg_features = self.eeg_normalization_layer(eeg_features)

        eeg_features = self.eeg_dimension_reduction(eeg_features)

        ''' Subject Encoder '''
        subject_features = self.subject_encoder(subject_info.float())
        if self.subject_normalization == 'CondBatchNorm':
            subject_features = self.subject_normalization_layer(subject_features, subject_id)
        elif self.subject_normalization == 'LayerNorm':
            subject_features = self.subject_normalization_layer(subject_features)
        subject_features = self.subj_dim_reduction(subject_features)

        ''' Conditioning '''
        subject_features = nn.functional.sigmoid(subject_features)
        x = torch.mul(subject_features, eeg_features)
        combined_features = self.cond_act(x)


        ''' Classify '''
        if self.combined_features_dimension:
            combined_features = self.final_linear(combined_features)
            combined_features = self.final_act(combined_features)
        combined_features = self.dropout(combined_features)
        output = self.classifier(combined_features)
        return output