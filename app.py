import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
from skimage.metrics import structural_similarity as ssim
from ConvLSTM import EnhancedConvLSTM
from PredRNN import PredRNN  # Add import for PredRNN
import json
import time


# Storage class remains the same
class PredictionStorage:
    def __init__(self, storage_path="predictions_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.video_path = self.storage_path / "videos"
        self.video_path.mkdir(exist_ok=True)
        self.data_path = self.storage_path / "data"
        self.data_path.mkdir(exist_ok=True)
        self.frames_path = self.storage_path / "frames"
        self.frames_path.mkdir(exist_ok=True)

    def save_prediction(self, predictions, metrics, timestamp, model_type):
        data = {
            'metrics': metrics,
            'timestamp': timestamp,
            'shape': predictions[0].shape,
            'num_frames': len(predictions),
            'model_type': model_type
        }
        with open(self.data_path / f"pred_{model_type}_{timestamp}.json", 'w') as f:
            json.dump(data, f)

        np.save(self.data_path / f"pred_{model_type}_{timestamp}.npy", np.array(predictions))

        frames_dir = self.frames_path / f"{model_type}_{timestamp}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(predictions):
            frame_path = frames_dir / f"pred_{i:03d}.jpg"
            frame_uint8 = (frame * 255).astype(np.uint8)
            cv2.imwrite(str(frame_path), frame_uint8)


def process_video(video_path, num_frames=10, color_mode='gray'):
    frames = []
    cap = cv2.VideoCapture(video_path)

    progress = st.progress(0)
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        if color_mode == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        progress.progress(len(frames) / num_frames)

    cap.release()

    if len(frames) < num_frames:
        st.warning(f"Video too short. Padding with last frame to reach {num_frames} frames.")
        last_frame = frames[-1] if frames else np.zeros((64, 64, 3) if color_mode == 'rgb' else (64, 64),
                                                        dtype=np.uint8)
        while len(frames) < num_frames:
            frames.append(last_frame.copy())

    return np.array(frames)


def predict_convlstm(model, input_frames, num_predictions=10):
    predictions = []
    current_input = input_frames

    progress = st.progress(0)
    with torch.no_grad():
        for i in range(num_predictions):
            pred = model(current_input)
            predictions.append(pred.squeeze().numpy())
            current_input = torch.cat([
                current_input[:, 1:],
                pred.unsqueeze(1)
            ], dim=1)
            progress.progress((i + 1) / num_predictions)

    return predictions


def predict_predrnn(model, input_frames, num_predictions=5):
    with torch.no_grad():
        predictions = model(input_frames, future_steps=num_predictions)
        return [pred.squeeze().numpy() for pred in predictions.unbind(1)]


def calculate_metrics(true_frames, pred_frames, multichannel=False):
    min_length = min(len(true_frames), len(pred_frames))
    true_frames = true_frames[:min_length]
    pred_frames = pred_frames[:min_length]

    mse = np.mean((true_frames - pred_frames) ** 2)

    # Modified SSIM calculation
    if multichannel:
        # For RGB images, calculate SSIM with channel_axis parameter
        ssim_score = ssim(true_frames.astype(np.float32),
                          pred_frames.astype(np.float32),
                          data_range=1.0,
                          channel_axis=2)  # Specify channel axis for RGB
    else:
        # For grayscale images
        ssim_score = ssim(true_frames.astype(np.float32),
                          pred_frames.astype(np.float32),
                          data_range=1.0,
                          multichannel=False)

    return mse, ssim_score


def create_prediction_video(frames, storage, timestamp, model_type, fps=4, is_color=False):
    video_path = storage.video_path / f"pred_{model_type}_{timestamp}.mp4"
    height, width = frames[0].shape[:2]

    frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]

    out = cv2.VideoWriter(str(video_path),
                          cv2.VideoWriter_fourcc(*'avc1'),
                          fps, (width, height),
                          is_color)

    if not out.isOpened():
        out = cv2.VideoWriter(str(video_path),
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height),
                              is_color)

    progress = st.progress(0)
    st.write("Creating video...")

    for i, frame in enumerate(frames_uint8):
        out.write(frame if is_color else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        progress.progress((i + 1) / len(frames))

    out.release()
    return str(video_path)


def main():
    st.title("Video Frame Prediction Models")

    storage = PredictionStorage()

    tab1, tab2 = st.tabs(["ConvLSTM", "PredRNN"])

    with tab1:
        st.header("Enhanced ConvLSTM Model")
        st.sidebar.info("ConvLSTM uses temporal attention and skip connections.")

        # Initialize ConvLSTM
        convlstm_model = EnhancedConvLSTM(
            input_channels=1,
            hidden_channels=64,
            kernel_size=3,
            num_layers=2
        )
        convlstm_model.load_state_dict(torch.load("best_convlstm_model.pth", map_location='cpu'))
        convlstm_model.eval()

        process_convlstm(convlstm_model, storage)

    with tab2:
        st.header("PredRNN Model")
        st.sidebar.info("PredRNN uses spatiotemporal memory cells for prediction.")

        # Initialize PredRNN
        predrnn_model = PredRNN(
            input_channels=3,
            hidden_channels=32,
            num_layers=4
        )
        predrnn_model.load_state_dict(torch.load("best_predrnn_model.pth", map_location='cpu'))
        predrnn_model.eval()

        process_predrnn(predrnn_model, storage)


def process_convlstm(model, storage):
    uploaded_file = st.file_uploader("Upload video (ConvLSTM)", type=['mp4', 'avi'])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())

            frames = process_video(temp_video.name, color_mode='gray')

            st.write("Input Frames:")
            cols = st.columns(min(5, len(frames)))
            for idx, col in enumerate(cols):
                col.image(frames[idx], caption=f"Frame {idx + 1}")

            if st.button("Predict with ConvLSTM"):
                timestamp = str(int(time.time()))

                input_tensor = torch.FloatTensor(frames).unsqueeze(1).unsqueeze(0) / 255.0

                with st.spinner("Generating predictions..."):
                    predictions = predict_convlstm(model, input_tensor)

                pred_array = np.array(predictions[:len(frames) - 1])
                true_array = frames[1:] / 255.0
                mse, ssim_score = calculate_metrics(true_array, pred_array)
                metrics = {'mse': float(mse), 'ssim': float(ssim_score)}

                col1, col2 = st.columns(2)
                col1.metric("Mean Squared Error", f"{mse:.6f}")
                col2.metric("SSIM Score", f"{ssim_score:.6f}")

                st.write("Predictions:")
                for i in range(0, len(predictions), 5):
                    pred_cols = st.columns(5)
                    for j, col in enumerate(pred_cols):
                        if i + j < len(predictions):
                            col.image((predictions[i + j] * 255).astype(np.uint8),
                                      caption=f"Prediction {i + j + 1}")

                video_path = create_prediction_video(predictions, storage, timestamp,
                                                     "convlstm", is_color=False)
                if video_path:
                    st.video(video_path)

                storage.save_prediction(predictions, metrics, timestamp, "convlstm")


def process_predrnn(model, storage):
    uploaded_file = st.file_uploader("Upload video (PredRNN)", type=['mp4', 'avi'])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())

            # Ensure frames are at least 64x64 for proper SSIM calculation
            frames = process_video(temp_video.name, color_mode='rgb')

            st.write("Input Frames:")
            cols = st.columns(min(5, len(frames)))
            for idx, col in enumerate(cols):
                col.image(frames[idx], caption=f"Frame {idx + 1}")

            if st.button("Predict with PredRNN"):
                timestamp = str(int(time.time()))

                # Prepare input tensor
                input_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2).unsqueeze(0) / 255.0

                with st.spinner("Generating predictions..."):
                    predictions = predict_predrnn(model, input_tensor)

                # Process predictions
                predictions = [p.transpose(1, 2, 0) for p in predictions]

                # Ensure frames are properly shaped for metrics
                pred_array = np.array(predictions[:len(frames) - 1])
                true_array = frames[1:] / 255.0

                # Calculate metrics with proper channel handling
                try:
                    mse, ssim_score = calculate_metrics(true_array, pred_array, multichannel=True)
                    metrics = {'mse': float(mse), 'ssim': float(ssim_score)}

                    col1, col2 = st.columns(2)
                    col1.metric("Mean Squared Error", f"{mse:.6f}")
                    col2.metric("SSIM Score", f"{ssim_score:.6f}")
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")
                    metrics = {'mse': None, 'ssim': None}

                # Display predictions
                st.write("Predictions:")
                for i in range(0, len(predictions), 5):
                    pred_cols = st.columns(5)
                    for j, col in enumerate(pred_cols):
                        if i + j < len(predictions):
                            pred_img = predictions[i + j]
                            col.image((pred_img * 255).astype(np.uint8),
                                      caption=f"Prediction {i + j + 1}")

                # Create and display video
                video_path = create_prediction_video(predictions,
                                                     storage, timestamp, "predrnn", is_color=True)
                if video_path:
                    st.video(video_path)

                # Save predictions
                storage.save_prediction(predictions, metrics, timestamp, "predrnn")


if __name__ == "__main__":
    main()