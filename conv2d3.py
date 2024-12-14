# 目前效果最好，lr = 0.0001,每10个epoch学习率乘0.1,试图增加噪声音量
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import soundfile as sf
import gc
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast  # 可选，用于混合精度训练

# 设置使用的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 请确保GPU编号存在，否则会导致无法使用GPU

# 获取当前文件夹中所有音频文件
def load_all_audio_files_in_folder(folder_path, file_extensions=("wav", "mp3")):
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(file_extensions)
    ]
    return audio_files

# 计算RMS值（均方根值），用于调整音量
def compute_rms(audio):
    return np.sqrt(np.mean(audio**2))

# 重建信号的函数
def reconstruct_signal(magnitude, phase, hop_length=512):
    complex_spectrum = magnitude * np.exp(1j * phase)
    return librosa.istft(complex_spectrum, hop_length=hop_length)

# 滑动窗口函数，用于创建样本
def create_sliding_windows(magnitude, window_size=20, hop_size=10):
    freq_bins, time_frames = magnitude.shape
    samples = []
    labels = []
    for start in range(0, time_frames - window_size + 1, hop_size):
        end = start + window_size
        sample = magnitude[:, start:end]  # (freq_bins, window_size)
        label = magnitude[:, start:end] * np.random.uniform(0.1, 0.3)  # 生成相关噪声标签
        samples.append(sample)
        labels.append(label)
    samples = np.array(samples)  # (num_samples, freq_bins, window_size)
    labels = np.array(labels)    # (num_samples, freq_bins, window_size)
    return samples, labels

# 读取并处理音频文件
def process_audio(file_path, n_fft=1024, hop_length=512, window_size=20, hop_size=10):
    audio, sr = librosa.load(file_path, sr=None)
    audio_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    audio_mag = np.abs(audio_stft)
    samples, labels = create_sliding_windows(audio_mag, window_size=window_size, hop_size=hop_size)
    return samples, labels, sr

# 定义2D CNN模型（包含批归一化）
class SpeechCancellingCNN2D(nn.Module):
    def __init__(self, input_channels, freq_bins, window_size):
        super(SpeechCancellingCNN2D, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((2, 2))  # 降维，每次池化将频率和时间各减半
        self.relu = nn.ReLU()

        # 计算经过卷积和池化后的特征维度
        self.feature_freq = freq_bins // 8  # 经过三次池化，每次频率减半
        self.feature_time = window_size // 8  # 经过三次池化，每次时间减半

        self.fc1 = nn.Linear(64 * self.feature_freq * self.feature_time, 1024)
        self.fc2 = nn.Linear(1024, freq_bins * window_size)

        # 保存频率和时间尺寸以便在 forward 中使用
        self.freq_bins = freq_bins
        self.window_size = window_size

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 16, freq_bins, window_size)
        x = self.pool(x)                        # (batch, 16, freq_bins/2, window_size/2)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 32, freq_bins/2, window_size/2)
        x = self.pool(x)                        # (batch, 32, freq_bins/4, window_size/4)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 64, freq_bins/4, window_size/4)
        x = self.pool(x)                        # (batch, 64, freq_bins/8, window_size/8)
        x = x.view(x.size(0), -1)               # 展平
        x = self.relu(self.fc1(x))              # (batch, 1024)
        x = self.fc2(x)                          # (batch, freq_bins * window_size)
        x = x.view(x.size(0), 1, self.freq_bins, self.window_size)  # (batch, 1, freq_bins, window_size)
        return x

# 定义保存模型的函数
def save_checkpoint(state, filename='model/model3_2d.pth'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Model saved to {filename}")

# 定义加载模型的函数
def load_model(checkpoint_path, input_channels, freq_bins, window_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpeechCancellingCNN2D(input_channels, freq_bins, window_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded model from epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss

# 读取所有音频文件
# folder_path = "./music"  # 请确保此路径正确，且包含音频文件
folder_path = "/SSD/22301126/coherent_noise/music_train"  # 请确保此路径正确，且包含音频文件
audio_files = load_all_audio_files_in_folder(folder_path)

audio_stfts = []
noise_stfts = []

n_fft = 1024
hop_length = 512
window_size = 20
hop_size = 10  # 步长

# 处理所有音频文件
for file_path in audio_files:
    try:
        samples, labels, sr = process_audio(file_path, n_fft, hop_length, window_size, hop_size)
        audio_stfts.append(samples)
        noise_stfts.append(labels)
        print(f"成功处理音频: {file_path}, 样本数量: {samples.shape[0]}")
    except Exception as e:
        print(f"处理音频 {file_path} 时出错: {e}")

# 确保有数据被加载
if not audio_stfts or not noise_stfts:
    raise ValueError("没有成功处理任何音频文件，或所有音频文件生成的样本不足。请检查音频文件。")

# 合并所有样本
features = np.concatenate(audio_stfts, axis=0)  # (total_num_samples, freq_bins, window_size)
labels = np.concatenate(noise_stfts, axis=0)    # (total_num_samples, freq_bins, window_size)

print(f"特征数量: {features.shape}, 标签数量: {labels.shape}")

# 确保有足够的样本
if features.shape[0] < 5:
    raise ValueError("样本数量不足（少于5个）。请增加音频文件数量或调整窗口参数。")

# 转换为PyTorch张量，并调整形状为 (batch, channels=1, freq_bins, window_size)
X = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # (num_samples, 1, freq_bins, window_size)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)    # (num_samples, 1, freq_bins, window_size)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")

# 将数据移动到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 16  # 根据GPU内存调整

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
freq_bins = X_train.shape[2]  # 例如，513
window_size = X_train.shape[3]  # 20

model = SpeechCancellingCNN2D(input_channels=1, freq_bins=freq_bins, window_size=window_size).to(device)
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # 可选：初始化GradScaler用于混合精度训练
# use_amp = True if device.type == 'cuda' else False
# if use_amp:
#     scaler = GradScaler()

import csv
import os

# Initialize CSV file for logging (create if it doesn't exist)
csv_filename = 'model/model3_2d.csv'
csv_fields = ['epoch', 'train_loss', 'val_loss']

# Check if the CSV file exists, if not, create it and write the header
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_fields)

# 训练循环
epochs = 1000000000000000000000000
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            val_outputs = model(batch_X)
            loss = criterion(val_outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # 每经过10个epoch，学习率乘以0.1
    if (epoch + 1) % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            print(f"Epoch {epoch + 1}: Learning rate adjusted to {param_group['lr']}")

    # 写入CSV文件
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, epoch_loss, val_loss])

    # 清理缓存
    del batch_X, batch_y, outputs, loss, val_outputs
    torch.cuda.empty_cache()
    gc.collect()

    # 保存模型如果验证损失更好
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, filename='model/model3_2d.pth')  # 修改这里，将 checkpoint_path 改为 filename


# 测试代码：加载模型并进行噪声预测
def load_model(checkpoint_path, input_channels, freq_bins, window_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpeechCancellingCNN2D(input_channels, freq_bins, window_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded model from epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss


# 计算音频的音量（RMS）
def adjust_audio_volume(audio, target_rms):
    current_rms = compute_rms(audio)
    scaling_factor = target_rms / current_rms if current_rms > 0 else 1.0
    return audio * scaling_factor


# 读取音频文件并处理
test_file_path = "music/0002.精选经典DJ - 丁当 - 我是一只小小鸟.wav"
test_samples, test_labels, sr = process_audio(test_file_path, n_fft=1024, hop_length=512, window_size=20, hop_size=10)

# 将测试样本转换为张量并调整形状
test_features = np.array(test_samples)  # (num_samples, freq_bins, window_size)
print("test_features shape1:", test_features.shape)
test_features = torch.tensor(test_features, dtype=torch.float32).unsqueeze(1).to(device)  # (num_samples, 1, freq_bins, window_size)
print('test_features.shape2', test_features.shape)

# 加载模型
checkpoint_path = "model/model3_2d.pth"
model, optimizer, epoch, best_loss = load_model(checkpoint_path, 1, test_features.shape[2], test_features.shape[3], device)

# 模型预测
model.eval()
with torch.no_grad():
    predicted_noise = model(test_features)

# 转换为numpy数组
predicted_noise = predicted_noise.squeeze(1).cpu().numpy()  # (num_samples, freq_bins, window_size)

print(f"Predicted noise shape: {predicted_noise.shape}")  # (num_samples, freq_bins, window_size)

# 计算原始音频的RMS值
original_audio, _ = librosa.load(test_file_path, sr=sr)
original_rms = compute_rms(original_audio)

# 重建完整的噪声幅度谱
num_samples = predicted_noise.shape[0]
freq_bins = predicted_noise.shape[1]
window_size = predicted_noise.shape[2]
hop_size = 10  # 与训练时相同的步长
total_time_frames = (num_samples - 1) * hop_size + window_size

# 初始化噪声幅度谱和计数器
full_noise_mag = np.zeros((freq_bins, total_time_frames))
count = np.zeros((freq_bins, total_time_frames))

for i in range(num_samples):
    start = i * hop_size
    end = start + window_size
    full_noise_mag[:, start:end] += predicted_noise[i]
    count[:, start:end] += 1

# 避免除以零
count[count == 0] = 1
full_noise_mag /= count

# 生成随机相位
phase = np.random.uniform(0, 2 * np.pi, size=full_noise_mag.shape)
complex_spectrum = full_noise_mag * np.exp(1j * phase)

# 重建噪声波形
reconstructed_noise = librosa.istft(complex_spectrum, hop_length=hop_length)

print("Reconstructed signal shape:", reconstructed_noise.shape)
print("Reconstructed signal dtype:", reconstructed_noise.dtype)

# 调整噪声音量与原始音频一致
adjusted_noise = adjust_audio_volume(reconstructed_noise, original_rms)

# 保存预测的噪声波形
os.makedirs("predict", exist_ok=True)
sf.write("predict/model3_2d.wav", adjusted_noise, sr)
print("Predicted noise saved as 'predict/model3_2d.wav'")
