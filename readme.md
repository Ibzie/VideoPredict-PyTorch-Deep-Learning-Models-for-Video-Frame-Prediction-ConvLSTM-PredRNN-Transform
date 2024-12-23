# Deep Learning Video Frame Prediction

This project implements deep learning models for video frame prediction using different architectures including ConvLSTM, PredRNN, and Transformer-based approaches. The models are trained on the UCF101 dataset and can predict future video frames based on a sequence of input frames.

## Project Structure

```
DEEP-LEARNING/
├── checkpoints/              # Model checkpoints directory
├── ucf101/                   # UCF101 dataset
│       ├── train/            # Training videos
│       │   ├── GolfSwing/
│       │   ├── PizzaTossing/
│       │   ├── Punch/
│       │   ├── Typing/
│       │   └── YoYo/
│       └── test/           # Testing videos
│           ├── GolfSwing/
│           ├── PizzaTossing/
│           ├── Punch/
│           ├── Typing/
│           └── YoYo/
├── app.py                   # Streamlit web application
├── ConvLSTM.py             # ConvLSTM model implementation
├── PredRNN.py              # PredRNN model implementation
├── Preprocessing.py        # Data preprocessing script
├── requirements.txt        # Project dependencies
├── Transformer.py          # Transformer model implementation
└── readme.md              # Project documentation
```

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.19.2
- opencv-python>=4.5.3
- scikit-image>=0.18.3
- matplotlib>=3.4.3
- streamlit>=1.0.0
- tqdm>=4.62.3
- Pillow>=8.3.2
- pytest>=6.2.5
- black>=21.9b0
- flake8>=3.9.2
- isort>=5.9.3

## Setup and Usage

### 1. Data Preparation

1. Download the UCF101 dataset and organize it in the following structure:
```
data/ucf101/
├── train/
│   ├── GolfSwing/
│   ├── PizzaTossing/
│   ├── Punch/
│   ├── Typing/
│   └── YoYo/
└── test/
    ├── GolfSwing/
    ├── PizzaTossing/
    ├── Punch/
    ├── Typing/
    └── YoYo/
```

2. Run the preprocessing script:
```bash
python Preprocessing.py --input-dir ucf101 --output-dir processed_data
```

Note: The preprocessing script is configured for Windows environments. For Linux users, file path separators and video reading mechanisms may need to be adjusted.

### 2. Training Models

Train each model separately using their respective Python files:

1. ConvLSTM Model:
```bash
python ConvLSTM.py
```

2. PredRNN Model:
```bash
python PredRNN.py
```

3. Transformer Model:
```bash
python Transformer.py
```

Each training script will:
- Load the preprocessed data
- Train the model
- Save the model checkpoints to the `checkpoints` directory

### 3. Running the Web Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

The web application will:
1. Allow you to select a model (ConvLSTM, PredRNN, or Transformer)
2. Upload a video or choose from sample videos
3. Generate and display frame predictions

## Model Descriptions

### ConvLSTM
- Convolutional LSTM architecture
- Combines spatial and temporal feature learning
- Suitable for capturing short-term motion patterns

### PredRNN
- Advanced spatiotemporal memory flow
- Multiple LSTM layers with skip connections
- Effective for long-term dependencies

### Transformer
- Self-attention mechanism for temporal modeling
- Position encoding for frame sequence information
- Memory-efficient implementation

## Troubleshooting

1. Video Loading Issues:
   - Ensure videos are in .avi or .mp4 format
   - Check file permissions
   - Verify OpenCV installation

2. CUDA/GPU Issues:
   - Verify PyTorch is installed with CUDA support
   - Check GPU memory usage
   - Adjust batch size if needed

3. Preprocessing Errors:
   - Ensure correct file paths
   - Verify disk space availability
   - Check input video format compatibility

## License

This project is open-source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request