import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, prediction_length=5):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.sequences = []
        self.action_classes = ['GolfSwing', 'PizzaTossing', 'Punch', 'Typing', 'YoYo']

        for action_class in self.action_classes:
            class_path = os.path.join(root_dir, action_class)
            if os.path.isdir(class_path):
                frames = sorted([f for f in os.listdir(class_path) if f.endswith('.jpg')])

                # Group frames by video ID
                video_ids = set('_'.join(f.split('_')[1:3]) for f in frames)

                for vid in video_ids:
                    video_frames = sorted([f for f in frames if vid in f])
                    if len(video_frames) >= sequence_length + prediction_length:
                        self.sequences.append((class_path, video_frames))

        logger.info(f"Found {len(self.sequences)} valid sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        class_path, frames = self.sequences[idx]

        # Random starting point
        max_start_idx = len(frames) - (self.sequence_length + self.prediction_length)
        start_idx = np.random.randint(0, max_start_idx + 1)

        # Load input sequence
        input_sequence = []
        for i in range(start_idx, start_idx + self.sequence_length):
            frame_path = os.path.join(class_path, frames[i])
            frame = Image.open(frame_path).convert('L')  # Convert to grayscale
            frame = self.transform(frame)
            input_sequence.append(frame)

        # Load target sequence
        target_sequence = []
        for i in range(start_idx + self.sequence_length,
                       start_idx + self.sequence_length + self.prediction_length):
            frame_path = os.path.join(class_path, frames[i])
            frame = Image.open(frame_path).convert('L')
            frame = self.transform(frame)
            target_sequence.append(frame)

        # Stack sequences and add channel dimension if needed
        input_tensor = torch.stack(input_sequence)  # [sequence_length, 1, 64, 64]
        target_tensor = torch.stack(target_sequence)  # [prediction_length, 1, 64, 64]

        return input_tensor, target_tensor


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        total_channels = 4 * self.hidden_channels

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=total_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_channels, self.hidden_channels, self.kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

        self.output_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels, height, width]
        batch_size, seq_len, _, height, width = x.size()

        hidden_state = self._init_hidden(batch_size, height, width)

        layer_output_list = []
        last_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x[:, t], [h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            x = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Generate prediction using the last hidden state
        predictions = self.output_conv(h)

        return predictions

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([
                torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device),
                torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
            ])
        return init_states


def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            # Reshape input: [batch, seq_len, channel, height, width]
            input_seq = input_seq.permute(0, 1, 2, 3, 4).to(device)
            target_seq = target_seq[:, 0].to(device)  # Take first target frame

            optimizer.zero_grad()
            predictions = model(input_seq)
            loss = criterion(predictions, target_seq)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq = input_seq.permute(0, 1, 2, 3, 4).to(device)
                target_seq = target_seq[:, 0].to(device)

                predictions = model(input_seq)
                loss = criterion(predictions, target_seq)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f'Epoch: {epoch}, Validation Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_convlstm_model.pth')
            logger.info(f'Saved best model with validation loss: {val_loss:.6f}')


def main():
    # Hyperparameters
    batch_size = 8
    sequence_length = 10
    prediction_length = 5
    hidden_channels = 64
    kernel_size = 3
    num_layers = 2
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = VideoFrameDataset('processed_data/train',
                                      sequence_length=sequence_length,
                                      prediction_length=prediction_length)
    val_dataset = VideoFrameDataset('processed_data/test',
                                    sequence_length=sequence_length,
                                    prediction_length=prediction_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ConvLSTM(
        input_channels=1,  # Grayscale images
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_layers=num_layers
    ).to(device)

    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs, device)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()