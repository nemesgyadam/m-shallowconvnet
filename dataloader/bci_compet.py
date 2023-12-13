from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np
import mne
from dataloader.augmentation import cutcat


def exponential_moving_standardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    r"""Perform exponential moving standardization.

    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.


    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T


class BCICompet2aIV(torch.utils.data.Dataset):
    def __init__(self, args):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        self.fs = args['sampling_rate']

        
        self.data, self.label = self.get_brain_data()

        self.calculate_subject_avg()
        self.calculate_subject_ftr()
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        if self.return_subject_info == 'id':
            sample = {'data': data, 'subject_info': self.target_subject, 'label': label}
        elif self.return_subject_info == 'avg':
            sample = {'data': data, 'subject_info': self.subject_avg, 'label': label}
        elif self.return_subject_info == 'ftr':
            sample = {'data': data, 'subject_info': self.subject_ftr, 'label': label}
        else:
            sample = {'data': data, 'label': label}
        
        return sample
    
    def calculate_subject_avg(self):
        self.subject_avg = np.mean(self.data, axis=0)
    
    def calculate_subject_ftr(self):
        BANDS = [(0,8), (8, 13), (13, 18), (18, 25), (25, 38)]
        sfreq = self.fs

        feature_values = []
        squeeze_data = np.squeeze(self.data)

        # Calculate features for each band
        for fmin, fmax in BANDS:
            psds, freqs = mne.time_frequency.psd_array_multitaper(squeeze_data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)

            average_power = np.mean(psds, axis=(1, 2))
            sum_power = np.sum(psds, axis=(1, 2))
            peak_frequency = freqs[np.argmax(psds, axis=2)].mean(axis=1)

            feature_values.append(np.mean(average_power))
            feature_values.append(np.mean(sum_power))
            feature_values.append(np.mean(peak_frequency))

        # Overall features across all bands
        psds, _ = mne.time_frequency.psd_array_multitaper(squeeze_data, sfreq=sfreq, fmin=8, fmax=25, verbose=False)
        overall_average_power = np.mean(psds)
        overall_sum_power = np.sum(psds)
        overall_std_dev_power = np.std(np.mean(psds, axis=(1, 2)))

        feature_values.extend([overall_average_power, overall_sum_power, overall_std_dev_power])

        self.subject_ftr = np.array(feature_values)

    def get_brain_data(self):

        filelist = sorted(glob(f'{self.base_path}/*T*.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E*.gdf'))
 
        
        label_filelist = sorted(glob(f'{self.base_path}/true_labels/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/true_labels/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if idx != self.target_subject: continue
                    
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)
            
            raw.load_data()
            raw.filter(0., 38., fir_design='firwin')
            raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0., 3.
            if not self.is_test:
                event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10}) if idx != 3 \
                else dict({'769': 5,'770': 6,'771': 7,'772': 8})
            else:
                event_id = dict({'783': 7})
            
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]
            
            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
                
        return data, label
    

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=8)
        
        return data, label
    
    
class BCICompet2bIV(torch.utils.data.Dataset):
    def __init__(self, args):
        '''
        * 769: left
        * 770: right
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T*.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E*.gdf'))
        
        label_filelist = sorted(glob(f'{self.base_path}/true_labels/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/true_labels/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if not self.is_test:
                if idx // 3 != self.target_subject: continue
            else:
                if idx // 2 != self.target_subject: continue
                        
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)

            raw.load_data()
            raw.filter(0., 38., fir_design='firwin')
            raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0., 3.
            if not self.is_test: event_id = dict({'769': annot['769'], '770': annot['770']})
            else: event_id = dict({'783': annot['783']})
                
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]

            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
        
        return data, label
            
    
    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        
        return data, label
    

def get_dataset(config_name, args):
    
    if 'bcicompet2a' in config_name:
        dataset = BCICompet2aIV(args)
    elif 'bcicompet2b' in config_name:
        dataset = BCICompet2bIV(args)
    else:
        raise Exception('get_dataset function Wrong dataset input!!!')
    
    return dataset

