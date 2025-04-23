import os
import numpy as np
from scipy.signal import welch, hilbert, resample 
from tqdm import tqdm,trange

class FeatureExtractor:

    # 频带定义
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'low-gamma': (30, 60), 'high-gamma': (60, 100)}
    cache_dir = './feature_cache'
    def __init__(self, p_1:list[np.array], p_2:list[np.array], samplerate=2000, down_samplerate=250,cached = False, n_segment= 0):
        assert len(p_1) == len(p_2), 'p1、p2含有的segment数量不相等'

        segment_time = []
        # 统一数据长度。这里我们认为采样率是相同的。
        for idx in range(len(p_1)):
            segment_len = min(p_1[idx].shape[1], p_2[idx].shape[1])
            p_1[idx] = p_1[idx][:segment_len]
            p_2[idx] = p_2[idx][:segment_len]
            segment_time.append(segment_len)
        segment_time = np.array(segment_time)
        self.y = (segment_time - np.min(segment_time)) / (np.max(segment_time) - np.min(segment_time))
        downsampled_p1 = []
        downsampled_p2 = []
        self.n_segment = len(p_1) if not cached else n_segment
        if not cached:
            for idx in trange(len(p_1),desc='downsampling',leave=True):
                downsampled_p1.append(resample(p_1[idx], int(p_1[idx].shape[1] / samplerate * down_samplerate), axis=1))
                downsampled_p2.append(resample(p_2[idx], int(p_2[idx].shape[1] / samplerate * down_samplerate), axis=1))

            self.p_1 = downsampled_p1
            self.p_2 = downsampled_p2
        
            for idx in range(self.n_segment):
                segment_len = min(self.p_1[idx].shape[1], self.p_2[idx].shape[1])
                self.p_1[idx] = self.p_1[idx][:,:segment_len]
                self.p_2[idx] = self.p_2[idx][:,:segment_len]

            self.samplerate = down_samplerate
        # 预先计算
        if not os.path.exists(FeatureExtractor.cache_dir):
            os.makedirs(FeatureExtractor.cache_dir)
        self.psd_cache = self._precompute_psd()
        self.plv_cache = self._precompute_plv()
    
    def _precompute_psd(self):
        psd_cache_file = os.path.join(FeatureExtractor.cache_dir, 'psd_cache.npy')
        if os.path.exists(psd_cache_file):
            return np.load(psd_cache_file, allow_pickle=True)
        else:
            psd_cache = []
            for segment in tqdm(self.p_1 + self.p_2, desc='precompute_psd', leave=True):
                segment_psd = []
                for channel_data in segment:
                    freqs, psd = welch(channel_data, fs=self.samplerate, nperseg=1024)
                    channel_psd = []
                    for band_name, band in FeatureExtractor.bands.items():
                        avg_psd = np.mean(psd[(freqs >= band[0]) & (freqs <= band[1])])
                        channel_psd.append(avg_psd)
                    segment_psd.append(channel_psd)
                psd_cache.append(segment_psd)
            np.save(psd_cache_file, psd_cache)
            return psd_cache
    
    def _precompute_plv(self):
        plv_cache_file = os.path.join(FeatureExtractor.cache_dir, 'plv_cache.npy')
        if os.path.exists(plv_cache_file):
            return np.load(plv_cache_file, allow_pickle=True)
        else:
            plv_cache = []
            for segment_p1, segment_p2 in tqdm(zip(self.p_1, self.p_2), desc='precompute_plv', leave=True, position=0):
                combined_segments = np.concatenate([segment_p1, segment_p2], axis=0)
                plv_matrix = np.zeros((len(combined_segments), len(combined_segments)))
                for i in trange(len(combined_segments), desc='segments', leave=True, position=1):
                    for j in range(i, len(combined_segments)):
                        phase_diff = np.angle(hilbert(combined_segments[i])) - np.angle(hilbert(combined_segments[j]))
                        plv = np.abs(np.sum(np.exp(1j * phase_diff)) / len(phase_diff))
                        plv_matrix[i, j] = plv
                        plv_matrix[j, i] = plv  # PLV是对称的
                plv_cache.append(plv_matrix)
            np.save(plv_cache_file, plv_cache)
            return plv_cache
    
    def get_PSD(self, channels=[]):
        psd_list = []
        for segment_psd in self.psd_cache:
            feature_vector = [segment_psd[channel] for channel in channels] + [segment_psd[channel+self.n_segment] for channel in channels]
            psd_list.append(np.array(feature_vector))
        return psd_list
    
    def get_PLV(self, channels=[]):
        plv_list = []
        channels = channels + [i + len(channels) for i in range(len(channels))]
        for plv_matrix in self.plv_cache:
            selected_plv_matrix = plv_matrix[np.ix_(channels, channels)]
            plv_list.append(selected_plv_matrix)
        return plv_list
    def get_Y(self):
        return self.y

def test():
    name1 = "ZCH"
    name2 = "LZ"
    file_dir = "./"
    p_1 = []
    p_2 = []
    files = os.listdir(file_dir)

    for filename in files:
        if filename.startswith(f"{name1}_") and filename.endswith(".npy"):
            filepath = os.path.join(file_dir, filename)
            data = np.load(filepath)
            p_1.append(data)

    for filename in files:
        if filename.startswith(f"{name2}_") and filename.endswith(".npy"):
            filepath = os.path.join(file_dir, filename)
            data = np.load(filepath)
            p_2.append(data)
    
    featureExtractor = FeatureExtractor(p_1,p_2,cached=True)

    from NN_train import RegressionOpti
    r_opti = RegressionOpti(8,6)
    channels = [0,1,2,3,4,5,6,7]
    psd =  featureExtractor.get_PSD(channels)
    plv = featureExtractor.get_PLV(channels)
    y = featureExtractor.get_Y()
    data = [(psd[i],plv[i],y[i]) for i in range(len(y))]
    import time
    t0 = time.time()
    result = r_opti.train_eval(data)
    print(result,time.time() - t0)

if __name__ == "__main__":
    test()