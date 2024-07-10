import os
import torch
import pandas as pd
import torchaudio
import cv2
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np

from .utils_dataset import get_transform

class BatvisionV2Dataset(Dataset):
    #处理batvision2中的数据
    def __init__(self, cfg, annotation_file, location_blacklist=None):
   
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format

        location_list = os.listdir(self.root_dir)
        if location_blacklist:
            location_list = [location for location in location_list if location not in location_blacklist]
        location_csv_paths = [os.path.join(self.root_dir, location, annotation_file) for location in location_list]
                
        self.instances = []
        
        for location_csv in location_csv_paths:
            self.instances.append(pd.read_csv(location_csv))
            
        self.instances = pd.concat(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # Access instance 
        instance = self.instances.iloc[idx]
        
        # 加载路径
        depth_path = os.path.join(self.root_dir,instance['depth path'],instance['depth file name'])
        audio_path = os.path.join(self.root_dir,instance['audio path'],instance['audio file name'])

        ## Depth
        # 加载深度图
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000 # to go from mm to m 从毫米到米
        if self.cfg.dataset.max_depth:
            depth[depth > self.cfg.dataset.max_depth] = self.cfg.dataset.max_depth 
        # 调整depth数据 转换成tensor类型  进行最小最大归一化
        depth_transform = get_transform(self.cfg, convert =  True, depth_norm = self.cfg.dataset.depth_norm)
        gt_depth = depth_transform(depth)
        
        ## Audio 
        # 加载声波 返回音频信号和采样率
        waveform, sr = torchaudio.load(audio_path)
        # STFT parameters for full length audio 音频傅里叶变换的参数
        win_length = 200 
        n_fft = 400
        hop_length = 100

        # Cut audio to fit max depth 最大长度切断音频
        if self.cfg.dataset.max_depth:
            #根据最大深度和采样率计算出截断值
            cut = int((2*self.cfg.dataset.max_depth / 340) * sr)
            #截断音频
            waveform = waveform[:,:cut]
            # Update STFT parameters 更新傅里叶变换的参数
            #窗口长度
            win_length = 64
            #傅里叶变换长度
            n_fft = 512
            #跳过长度
            hop_length=64//4

        # Process sound
        # 如果音频格式为spectrogram
        if 'spectrogram' in self.audio_format:
            # 如果音频格式为mel，则获取梅尔频谱
            if 'mel' in self.audio_format:
                spec = self._get_melspectrogram(waveform, n_fft = n_fft, power = 1.0, win_length = win_length)
            else:
                # 如果音频格式为普通频谱，则获取普通频谱
                spec = self._get_spectrogram(waveform, n_fft = n_fft, power = 1.0, win_length = win_length, hop_length =  hop_length)
            # 获取频谱变换
            spec_transform =  get_transform(self.cfg, convert = False) # convert False因为已经是tensor
            # 对频谱进行变换
            audio2return = spec_transform(spec)
        # 如果音频格式为waveform
        elif 'waveform' in self.audio_format:
            # 直接返回波形
            audio2return = waveform
        
        return{"audio_spec":audio2return,"depth": gt_depth,"audio_wave":waveform}
    
    # audio transformation: spectrogram获取频谱图
    def _get_spectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, hop_length=100): 

        spectrogram = T.Spectrogram(
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          hop_length=hop_length,
        )
        #db = T.AmplitudeToDB(stype = 'magnitude')
        return spectrogram(waveform)
    
    # audio transformation: mel spectrogram
    def _get_melspectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, f_min = 20.0, f_max = 20000.0): 

        melspectrogram = T.MelSpectrogram(sample_rate = 44100, 
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          f_min = f_min, 
          f_max = f_max,
          n_mels = 32, 
        )
        return melspectrogram(waveform)
    