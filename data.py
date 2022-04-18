import math
import random
import os
from pathlib import Path
from functools import partial

import numpy as np
from lfcc import lfcc

import torch
import torchaudio
from torch.utils.data import Dataset

#we provide two kinds of dataset here, one based on the LFCC features obtained from the Matlab Baseline of ASVspoof 2019, 
#the other based on the raw audio and compute the LFCC feature in Python
#these LFCC features are different, the results in out paper are all based on the former one
def pad_audio(waveform, length):
    frames = waveform.shape[-1]
    if frames < 16000*length:
        n = 16000*length//frames+1
        waveform = torch.tile(waveform, (1, n))[:, :16000*length]
    elif frames > 16000*length:
        n = frames-16000*length
        index_start = np.random.randint(low=0, high=n+1)
        waveform = waveform[:, index_start:index_start+16000*length]

    return waveform


def pad_feature(feature):
    frames=feature.shape[0]
    if frames<512:
        n=512//frames+1
        feature=np.tile(feature, (n, 1))[:512, :]
    elif frames>512:
        n=frames-512
        index_start=np.random.randint(low=0, high=n+1)
        feature=feature[index_start:index_start+512, :]
    return feature


feature_extractor=partial(lfcc, win_len=0.02, win_hop=0.01)


class AudioPhase2(Dataset):
    def __init__(self, audio_path, protocol_path, length):
        self.audios = audio_path
        self.length = length
        with open(protocol_path) as file:
            meta_infos = file.readlines()
        self.meta_infos = meta_infos
        self.mapping = {meta_info.replace('\n', '').split(' ')[1]: meta_info.replace(
            '\n', '').split(' ')[-1] for meta_info in meta_infos}

    def __getitem__(self, index):
        info = self.meta_infos[index]
        name = info.replace('\n', '').split(' ')[1]
        second_path=info.replace('\n', '').split(' ')[0]
        path = os.path.join(self.audios, second_path, name+'.wav')
        waveform, _ = torchaudio.load(path)

        waveform_info = self.mapping[name]
        target = 1 if waveform_info == 'spoof' else 0
        waveform = pad_audio(waveform, length=self.length)
        waveform_lfcc = lfcc(waveform.T.numpy(), win_len=0.02, win_hop=0.01)        
        return waveform_lfcc, target, name
    
    def __len__(self):
        return len(self.meta_infos)


class Phase11(Dataset):
    def __init__(self, genuine_path, spoof_path, protocol_path):
        self.genuine=list(Path(genuine_path).glob('*.npz'))
        self.spoof=list(Path(spoof_path).glob('*.npz'))
        with open(protocol_path) as file:
            meta_infos=file.readlines()
        self.mapping={meta_info.replace('\n', '').split(' ')[1]:meta_info.replace('\n', '').split(' ') for meta_info in meta_infos}
        self.features=self.genuine+self.spoof
    def __getitem__(self, index):
        feature_path=self.features[index]
        feature_info=self.mapping[feature_path.name.replace('.npz', '')]
        ways=feature_info[-2]
        target=0 if feature_info[-1]=='bonafide' else 1
        feature=np.load(feature_path)['arr_0']
        feature=pad_feature(feature)
        return feature, target, feature_info
    def __len__(self):
        return len(self.features)