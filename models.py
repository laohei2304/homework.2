import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        low_hz = torch.linspace(0, sample_rate / 2 - min_band_hz, out_channels)
        high_hz = low_hz + min_band_hz + torch.abs(torch.randn(out_channels) * sample_rate / (2 * out_channels))
        self.low_hz_ = nn.Parameter(low_hz)
        self.band_hz_ = nn.Parameter(high_hz - low_hz)
        h = kernel_size // 2
        n = torch.arange(-h, h + 1, dtype=torch.float32)
        self.register_buffer("n_", n)
        self.register_buffer("window_", torch.hamming_window(kernel_size, periodic=False))
    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), max=self.sample_rate / 2)
        band = high - low
        low_rad = 2 * math.pi * low / self.sample_rate
        high_rad = 2 * math.pi * high / self.sample_rate
        n_safe = torch.where(self.n_ == 0, torch.ones_like(self.n_), self.n_)
        diff = (torch.sin(high_rad[:, None] * self.n_[None, :]) - torch.sin(low_rad[:, None] * self.n_[None, :])) / (math.pi * n_safe[None, :])
        diff = torch.where(self.n_[None, :] == 0, (2 * band / self.sample_rate)[:, None], diff)
        filters = (diff * self.window_[None, :])
        filters = filters / (filters.abs().sum(dim=1, keepdim=True) + 1e-8)
        return F.conv1d(x, filters.view(self.out_channels, 1, self.kernel_size), stride=1, padding=self.kernel_size // 2)

class SincNetTransformer(nn.Module):
    def __init__(self, sample_rate=16000, num_machines=4):
        super().__init__()
        self.sinc_conv = SincConv(80, 251, sample_rate)
        self.bn1 = nn.BatchNorm1d(80)
        self.pool1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(80, 60, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(60)
        self.pool2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(60)
        self.pool3 = nn.MaxPool1d(3, stride=3)
        self.conv4 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(60)
        self.pool4 = nn.MaxPool1d(3, stride=3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=60, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.machine_head = nn.Sequential(nn.Linear(60, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, num_machines))
        self.anomaly_head = nn.Sequential(nn.Linear(60, 64), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, 2))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.sinc_conv(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.transformer(x.permute(0, 2, 1))
        feat = x.mean(dim=1)
        return self.machine_head(feat), self.anomaly_head(feat)