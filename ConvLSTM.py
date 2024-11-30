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
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_channels):
        super(TemporalAttention, self).__init__()
        self.query = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.key = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.value = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels, height, width]
        batch_size, seq_len, channels, height, width = x.size()

        # Reshape for attention computation
        x_flat = x.view(batch_size, seq_len, -1)  # [B, T, C*H*W]

        # Compute query, key, value
        q = self.query(x.view(-1, channels, height, width)).view(batch_size, seq_len, -1)  # [B, T, C*H*W]
        k = self.key(x.view(-1, channels, height, width)).view(batch_size, seq_len, -1)  # [B, T, C*H*W]
        v = self.value(x.view(-1, channels, height, width)).view(batch_size, seq_len, -1)  # [B, T, C*H*W]

        # Compute attention scores
        attention = torch.bmm(q, k.transpose(1, 2))  # [B, T, T]
        attention = F.softmax(attention / np.sqrt(channels * height * width), dim=2)

        # Apply attention
        out = torch.bmm(attention, v)  # [B, T, C*H*W]
        out = out.view(batch_size, seq_len, channels, height, width)

        return self.gamma * out + x

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
    def __init__(self, input_channels, hidden_channels, kernel_size, dropout=0.3):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.dropout = nn.Dropout2d(dropout)

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.dropout(combined)
        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# class ConvLSTM(nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
#         super(ConvLSTM, self).__init__()
#
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#
#         cell_list = []
#         for i in range(num_layers):
#             cur_input_channels = input_channels if i == 0 else hidden_channels
#             cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size))
#         self.cell_list = nn.ModuleList(cell_list)
#
#         self.motion_encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#
#         self.decoder = nn.Sequential(
#             nn.Conv2d(hidden_channels + 64, hidden_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_channels),
#             nn.ReLU(inplace=True),
#             ResidualBlock(hidden_channels),
#             ResidualBlock(hidden_channels),
#             nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
#
#     def compute_motion_features(self, x):
#         # Reshape input for motion computation
#         b, t, c, h, w = x.size()
#         x_reshaped = x.view(b * t, c, h, w)
#
#         # Compute temporal differences
#         diffs = x_reshaped[1:] - x_reshaped[:-1]
#         motion_feature = diffs.mean(dim=0, keepdim=True)
#
#         return self.motion_encoder(motion_feature)
#
#     def forward(self, x):
#         batch_size, seq_len, _, height, width = x.size()
#         hidden_state = self._init_hidden(batch_size, height, width)
#
#         motion_features = self.compute_motion_features(x)
#
#         for layer_idx in range(self.num_layers):
#             h, c = hidden_state[layer_idx]
#             output_inner = []
#
#             for t in range(seq_len):
#                 h, c = self.cell_list[layer_idx](x[:, t], [h, c])
#                 output_inner.append(h)
#
#             layer_output = torch.stack(output_inner, dim=1)
#             x = layer_output
#
#         predictions = self.decoder(torch.cat([h, motion_features.expand(batch_size, -1, height, width)], dim=1))
#         return predictions
#
#     def _init_hidden(self, batch_size, height, width):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append([
#                 torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device),
#                 torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
#             ])
#         return init_states


class EnhancedConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(EnhancedConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # ConvLSTM cells
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

        # Temporal attention module
        self.temporal_attention = TemporalAttention(hidden_channels)

        # Motion encoder
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )

        # Decoder - Fixed dimensions
        decoder_in_channels = hidden_channels + 64  # Last hidden state + motion features
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def compute_motion_features(self, x):
        # x shape: [batch_size, seq_len, channels, height, width]
        b, t, c, h, w = x.size()
        x_reshaped = x.view(b * t, c, h, w)

        # Compute temporal differences
        diffs = x_reshaped[1:] - x_reshaped[:-1]
        motion_feature = diffs.mean(dim=0, keepdim=True)

        return self.motion_encoder(motion_feature)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debug print
        batch_size, seq_len, channels, height, width = x.size()
        hidden_state = self._init_hidden(batch_size, height, width)

        # Compute motion features
        motion_features = self.compute_motion_features(x)
        print(f"Motion features shape: {motion_features.shape}")  # Debug print

        # Process sequence with ConvLSTM cells
        layer_output = None
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x[:, t] if layer_idx == 0 else layer_output[:, t], [h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            print(f"Layer {layer_idx} output shape: {layer_output.shape}")  # Debug print

        # Apply temporal attention
        attended_features = self.temporal_attention(layer_output)
        print(f"Attended features shape: {attended_features.shape}")  # Debug print

        # Take last timestep
        h = attended_features[:, -1]
        print(f"Last hidden state shape: {h.shape}")  # Debug print

        # Concatenate with motion features
        decoder_input = torch.cat([h, motion_features.expand(batch_size, -1, height, width)], dim=1)
        print(f"Decoder input shape: {decoder_input.shape}")  # Debug print

        # Final prediction
        predictions = self.decoder(decoder_input)
        print(f"Predictions shape: {predictions.shape}")  # Debug print

        return predictions

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([
                torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device),
                torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
            ])
        return init_states


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq = input_seq.to(device)
            target = target_seq[:, 0].to(device)  # First frame of target sequence

            optimizer.zero_grad()
            predictions = model(input_seq)
            loss = criterion(predictions, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        val_loss = validate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(f'Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_convlstm_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            break


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq = input_seq.to(device)
            target = target_seq[:, 0].to(device)
            predictions = model(input_seq)
            val_loss += criterion(predictions, target).item()
    return val_loss / len(val_loader)


def main():
    batch_size = 8
    sequence_length = 10
    prediction_length = 5
    hidden_channels = 64
    kernel_size = 3
    num_layers = 2
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_dataset = VideoFrameDataset('processed_data/train',
                                    sequence_length=sequence_length,
                                    prediction_length=prediction_length)
    val_dataset = VideoFrameDataset('processed_data/test',
                                  sequence_length=sequence_length,
                                  prediction_length=prediction_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model = ConvLSTM(
    #     input_channels=1,
    #     hidden_channels=hidden_channels,
    #     kernel_size=kernel_size,
    #     num_layers=num_layers
    # ).to(device)

    model = EnhancedConvLSTM(
        input_channels=1,  # For grayscale images
        hidden_channels=64,  # Hidden state size
        kernel_size=3,  # Convolution kernel size
        num_layers=2  # Number of ConvLSTM layers
    ).to(device)

    train_model(model, train_loader, val_loader, num_epochs, device)


if __name__ == "__main__":
    main()