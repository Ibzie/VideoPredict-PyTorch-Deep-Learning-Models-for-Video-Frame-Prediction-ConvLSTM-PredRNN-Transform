import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from pathlib import Path


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class FrameEmbedding(nn.Module):
    def __init__(self, input_channels, d_model):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(d_model * 8 * 8, d_model)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(b * s, self.d_model * 8 * 8)
        x = self.proj(x)
        return x.view(b, s, self.d_model)


class RecurrentMemoryTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_memory_tokens=8):
        super().__init__()
        self.d_model = d_model
        self.num_memory_tokens = num_memory_tokens

        self.frame_embedding = FrameEmbedding(input_channels=3, d_model=d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.memory_tokens = nn.Parameter(torch.randn(1, num_memory_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder_linear = nn.Sequential(
            nn.Linear(d_model, 64 * 8 * 8),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        x = self.frame_embedding(x)

        if prev_memory is None:
            prev_memory = self.memory_tokens.expand(batch_size, -1, -1)

        x = torch.cat([prev_memory, x, prev_memory], dim=1)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        new_memory = x[-self.num_memory_tokens:].transpose(0, 1)
        x = x[self.num_memory_tokens:-self.num_memory_tokens]
        x = x.transpose(0, 1)

        # Take only the last 5 positions
        x = x[:, -5:, :]  # Match target sequence length

        b, s, d = x.shape
        x = self.decoder_linear(x.reshape(b * s, -1))
        x = x.view(b * s, 64, 8, 8)
        x = self.decoder_conv(x)
        x = x.view(b, s, 3, 64, 64)

        return x, new_memory

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, prediction_length=5, transform=None, max_samples_per_category=100):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        self.samples = []
        for category in os.listdir(root_dir):
            category_path = Path(root_dir) / category
            if not category_path.is_dir():
                continue

            frame_files = sorted([str(f) for f in category_path.glob("*.jpg")])

            if len(frame_files) >= sequence_length + prediction_length:
                step = max(1, (len(frame_files) - (sequence_length + prediction_length)) // max_samples_per_category)
                for i in range(0, len(frame_files) - (sequence_length + prediction_length) + 1, step):
                    self.samples.append(frame_files[i:i + sequence_length + prediction_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames = []
        for frame_path in self.samples[idx]:
            frame = Image.open(frame_path).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
        sequence = torch.stack(frames)
        return sequence[:self.sequence_length], sequence[self.sequence_length:]


def train_model(model, train_loader, val_loader, num_epochs, device, save_dir):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (input_frames, target_frames) in enumerate(progress_bar):
            input_frames, target_frames = input_frames.to(device), target_frames.to(device)

            optimizer.zero_grad()
            try:
                with torch.cuda.amp.autocast():
                    output_frames, _ = model(input_frames)
                    if output_frames.shape != target_frames.shape:
                        raise ValueError(
                            f"Shape mismatch: output {output_frames.shape} vs target {target_frames.shape}")
                    loss = criterion(output_frames, target_frames)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                progress_bar.set_postfix({
                    'train_loss': f'{train_loss / (batch_idx + 1):.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            except Exception as e:
                logging.error(f"Forward pass failed at batch {batch_idx}: {str(e)}")
                continue

            if (batch_idx + 1) % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_dir / 'latest_checkpoint.pth')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_frames, target_frames in val_loader:
                input_frames, target_frames = input_frames.to(device), target_frames.to(device)
                try:
                    with torch.cuda.amp.autocast():
                        output_frames, _ = model(input_frames)
                        if output_frames.shape != target_frames.shape:
                            raise ValueError(
                                f"Shape mismatch: output {output_frames.shape} vs target {target_frames.shape}")
                        val_loss += criterion(output_frames, target_frames).item()
                except Exception as e:
                    logging.error(f"Validation forward pass failed: {str(e)}")
                    continue

        val_loss /= len(val_loader)
        logging.info(f'Epoch {epoch + 1}, Val Loss: {val_loss:.6f}')
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_dir / 'best_model.pth')


def visualize_predictions(model, test_loader, device, save_dir, num_samples=5):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i, (input_frames, target_frames) in enumerate(test_loader):
            if i >= num_samples:
                break

            input_frames = input_frames.to(device)
            predictions, _ = model(input_frames)

            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for t in range(5):
                axes[0, t].imshow(input_frames[0, -1].cpu().permute(1, 2, 0))
                axes[0, t].axis('off')
                axes[0, t].set_title('Input')

                axes[1, t].imshow(target_frames[0, t].cpu().permute(1, 2, 0))
                axes[1, t].axis('off')
                axes[1, t].set_title('Target')

                axes[2, t].imshow(predictions[0, t].cpu().permute(1, 2, 0))
                axes[2, t].axis('off')
                axes[2, t].set_title('Predicted')

            plt.tight_layout()
            plt.savefig(save_dir / f'prediction_sample_{i}.png')
            plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    dataset = VideoFrameDataset("processed_data/train", sequence_length=10, prediction_length=5)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = RecurrentMemoryTransformer().to(device)
    train_model(model, train_loader, val_loader, num_epochs=100, device=device, save_dir=save_dir)
    visualize_predictions(model, val_loader, device, save_dir / "visualizations")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Training failed with error:")
        raise