import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import random_split
import torch.backends.cudnn as cudnn

# Custom Dataset for UCF101 frames because other methods were not working at alllllllll

#For Ubuntu
# class UCF101FramesDataset(Dataset):
#     def __init__(self, root_dir, sequence_length=10, prediction_length=5):
#         self.root_dir = root_dir
#         self.sequence_length = sequence_length
#         self.prediction_length = prediction_length
#         self.samples = []
#
#         # Get all action categories
#         categories = os.listdir(root_dir)
#         for category in categories:
#             category_path = os.path.join(root_dir, category)
#             if not os.path.isdir(category_path):
#                 continue
#
#             videos = os.listdir(category_path)
#             for video in videos:
#                 frames = sorted([f for f in os.listdir(os.path.join(category_path, video))
#                                  if f.endswith('.jpg')])
#
#                 if len(frames) >= sequence_length + prediction_length:
#                     self.samples.append((category_path, video, frames))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         category_path, video, frames = self.samples[idx]
#
#         sequence = []
#         for i in range(self.sequence_length + self.prediction_length):
#             frame_path = os.path.join(category_path, video, frames[i])
#             frame = Image.open(frame_path).convert('RGB')
#             frame = transforms.Resize((64, 64))(frame)
#             frame = transforms.ToTensor()(frame)
#             sequence.append(frame)
#
#         sequence = torch.stack(sequence)
#         return sequence[:self.sequence_length], sequence[self.sequence_length:]

#For Windows

class UCF101FramesDataset(Dataset):
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
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue

            frame_files = sorted([
                os.path.join(category_path, f)
                for f in os.listdir(category_path)
                if f.endswith('.jpg')
            ])

            if len(frame_files) >= sequence_length + prediction_length:
                # Take fewer samples per category
                step = max(1, (len(frame_files) - (sequence_length + prediction_length)) // max_samples_per_category)
                for i in range(0, len(frame_files) - (sequence_length + prediction_length) + 1, step):
                    self.samples.append(frame_files[i:i + sequence_length + prediction_length])

        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frames = []
        for frame_path in self.samples[idx]:
            frame = Image.open(frame_path).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)

        sequence = torch.stack(frames)
        return sequence[:self.sequence_length], sequence[self.sequence_length:]

class STLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(STLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels

        # Input, forget, cell, output gates
        self.conv_gates = nn.Conv2d(
            in_channels + 2 * hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1
        )

        # Spatiotemporal memory gate
        self.conv_m = nn.Conv2d(
            in_channels + 2 * hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, h_prev, c_prev, m_prev):
        combined = torch.cat([x, h_prev, m_prev], dim=1)
        gates = self.conv_gates(combined)

        # Split gates
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)

        # Update cell state
        c_next = f * c_prev + i * g

        # Update spatiotemporal memory
        m_next = torch.sigmoid(self.conv_m(combined)) * torch.tanh(c_next)

        # Update hidden state
        h_next = o * torch.tanh(c_next + m_next)

        return h_next, c_next, m_next


class PredRNN(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, num_layers=4):
        super(PredRNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.st_lstm_layers = nn.ModuleList([
            STLSTMCell(
                in_channels=input_channels if l == 0 else hidden_channels,
                hidden_channels=hidden_channels
            )
            for l in range(num_layers)
        ])

        self.output_conv = nn.Conv2d(hidden_channels, input_channels, 1)

    def init_hidden(self, batch_size, height, width):
        device = next(self.parameters()).device
        hidden_states = []
        for _ in range(self.num_layers):
            hidden_states.append((
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            ))
        return hidden_states

    def forward(self, input_sequence, future_steps):
        batch_size, seq_length, channels, height, width = input_sequence.size()
        hidden_states = self.init_hidden(batch_size, height, width)
        outputs = []

        # Process input sequence
        for t in range(seq_length):
            x = input_sequence[:, t]
            new_hidden_states = []
            for l in range(self.num_layers):
                h, c, m = hidden_states[l]
                h_new, c_new, m_new = self.st_lstm_layers[l](
                    x if l == 0 else new_hidden_states[-1][0],
                    h, c, m
                )
                new_hidden_states.append((h_new, c_new, m_new))
            hidden_states = new_hidden_states

        # Generate future predictions
        for t in range(future_steps):
            x = self.output_conv(hidden_states[-1][0])
            new_hidden_states = []
            for l in range(self.num_layers):
                h, c, m = hidden_states[l]
                h_new, c_new, m_new = self.st_lstm_layers[l](
                    x if l == 0 else new_hidden_states[-1][0],
                    h, c, m
                )
                new_hidden_states.append((h_new, c_new, m_new))
            hidden_states = new_hidden_states
            outputs.append(x)

        return torch.stack(outputs, dim=1)


def train_model(model, train_loader, val_loader, num_epochs=10, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (input_frames, target_frames) in enumerate(progress_bar):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_frames, target_frames.size(1))
                loss = criterion(outputs, target_frames)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': train_loss / (batch_idx + 1)})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_frames, target_frames in val_loader:
                input_frames = input_frames.to(device)
                target_frames = target_frames.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(input_frames, target_frames.size(1))
                    val_loss += criterion(outputs, target_frames).item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_predrnn_model.pth')


def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        for i, (input_frames, target_frames) in enumerate(test_loader):
            if i >= num_samples:
                break

            input_frames = input_frames.to(device)
            predictions = model(input_frames, target_frames.size(1))

            # Plot results
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for t in range(5):
                # Input sequence (last frame)
                axes[0, t].imshow(input_frames[0, -1].cpu().permute(1, 2, 0))
                axes[0, t].axis('off')
                axes[0, t].set_title('Input')

                # Target frame
                axes[1, t].imshow(target_frames[0, t].cpu().permute(1, 2, 0))
                axes[1, t].axis('off')
                axes[1, t].set_title('Target')

                # Predicted frame
                axes[2, t].imshow(predictions[0, t].cpu().permute(1, 2, 0))
                axes[2, t].axis('off')
                axes[2, t].set_title('Predicted')

            plt.tight_layout()
            plt.savefig(f'prediction_sample_{i}.png')
            plt.close()


# Usage example:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UCF101FramesDataset("processed_data/train",
                                  sequence_length=10,
                                  prediction_length=5,
                                  max_samples_per_category=100)  # Limit samples

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced batch size
        shuffle=True,
        num_workers=2,  # Reduced workers
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = PredRNN(input_channels=3, hidden_channels=32).to(device)  # Reduced hidden channels
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)