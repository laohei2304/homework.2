import os
import random
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

def scan_all_data(root_dir):
    all_files, classes = [], set()
    for dirpath, _, filenames in os.walk(root_dir):
        parts = dirpath.split(os.sep)
        if any(p in parts for p in ['normal', '0_normal', 'abnormal', '1_abnormal']):
            machine = "unknown"
            for i, p in enumerate(parts):
                if p in ['normal', '0_normal', 'abnormal', '1_abnormal']:
                    for j in range(i-1, -1, -1):
                        cand = parts[j]
                        if cand in ['train', 'test', '6_dB', '0_dB', '-6_dB', '-6_db'] or cand.startswith('id_'):
                            continue
                        machine = cand
                        break
                    break
            if machine != "unknown":
                classes.add(machine)
                for f in filenames:
                    if f.lower().endswith(('.wav', '.flac')):
                        all_files.append(os.path.join(dirpath, f))
    return sorted(list(classes)), all_files

class MIMIIMultiTaskDataset(Dataset):
    def __init__(self, filepaths, class_names, sample_rate=16000, duration=4.0, is_train=True):
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.is_train = is_train
        self.filepaths, self.machine_labels, self.anomaly_labels = [], [], []
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        for fpath in filepaths:
            parts = fpath.split(os.sep)
            machine = next((c for c in class_names if c in parts), "unknown")
            if machine == "unknown": continue
            if any(p in parts for p in ["normal", "0_normal"]): label = 0
            elif any(p in parts for p in ["abnormal", "1_abnormal"]): label = 1
            else: continue
            self.filepaths.append(fpath)
            self.machine_labels.append(class_to_idx[machine])
            self.anomaly_labels.append(label)
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        data, sr = sf.read(self.filepaths[idx], dtype='float32', always_2d=True)
        waveform = torch.from_numpy(data.T).contiguous()
        if sr != self.sample_rate: waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True).squeeze(0)
        if waveform.size(0) < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - waveform.size(0)))
        else:
            if self.is_train:
                start = torch.randint(0, waveform.size(0) - self.num_samples + 1, (1,)).item()
                waveform = waveform[start:start + self.num_samples]
                waveform = waveform * random.uniform(0.8, 1.2) + torch.randn_like(waveform) * 0.005
            else:
                start = (waveform.size(0) - self.num_samples) // 2
                waveform = waveform[start:start + self.num_samples]
        return waveform, self.machine_labels[idx], self.anomaly_labels[idx]