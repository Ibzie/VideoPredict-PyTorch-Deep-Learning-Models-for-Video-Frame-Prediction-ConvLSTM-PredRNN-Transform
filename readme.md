# Deep Learning Video Frame Prediction

This project implements three different deep learning models for video frame prediction using the UCF101 dataset. The models are capable of generating future frames based on a sequence of input frames, effectively predicting how an action will continue.

## Models Implemented

1. **Enhanced ConvLSTM**: A Convolutional LSTM model with temporal attention mechanism
2. **PredRNN**: Implementation of PredRNN with spatiotemporal memory
3. **Recurrent Memory Transformer**: A transformer-based approach for video prediction

## Dataset

The project uses the UCF101 Action Recognition Dataset, focusing on five specific action categories:
- YoYo
- Typing
- Punch
- PizzaTossing
- GolfSwing

Download the dataset from: [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

## Project Structure

```
Deep-Learning-Project/
├── data/
│   ├── ucf101/                  # Raw dataset
│   └── processed_data/          # Preprocessed frames
├── models/
│   ├── convlstm.py             # Enhanced ConvLSTM implementation
│   ├── predrnn.py              # PredRNN implementation
│   └── transformer.py          # Recurrent Memory Transformer
├── utils/
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── evaluation.py           # Evaluation metrics
├── config/
│   └── config.yaml             # Configuration parameters
├── requirements.txt            # Project dependencies
└── notebooks/                  # Jupyter notebooks for analysis
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Download and extract the UCF101 dataset to the `data/ucf101` directory

4. Preprocess the data:
```bash
python utils/preprocessing.py
```

## Model Training

Each model can be trained separately using their respective training scripts:

```bash
python train.py --model convlstm  # For Enhanced ConvLSTM
python train.py --model predrnn   # For PredRNN
python train.py --model transformer  # For Recurrent Memory Transformer
```

## Performance

Model performance metrics on different action sequences:

### Typing Action Sequence
| Model | MSE | SSIM |
|-------|-----|------|
| Enhanced ConvLSTM | 0.099558 | 0.296273 |
| PredRNN | 0.019115 | 0.785814 |
| Recurrent Memory Transformer | 0.034680 | 0.354711 |

### YoYo Action Sequence
| Model | MSE | SSIM |
|-------|-----|------|
| Enhanced ConvLSTM | 0.020076 | 0.372000 |
| PredRNN | 0.040034 | 0.758295 |
| Recurrent Memory Transformer | 0.022363 | 0.456513 |

## Implementation Features

- Gradient accumulation for efficient training
- Mixed precision training using torch.cuda.amp
- Custom temporal consistency loss function
- Lazy loading dataset implementation
- Memory-efficient backpropagation
- Dynamic batch size adjustment
- Cyclical learning rates

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA capable GPU (recommended)
- Other dependencies listed in requirements.txt

## Linux Platform Specific Instructions

### System Requirements
- OpenCV dependencies: `sudo apt-get install libgl1-mesa-glx`
- FFmpeg for video processing: `sudo apt-get install ffmpeg`

### Installation Steps
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Known Issues and Solutions

1. **Video Codec Support**
   - The application tries multiple codecs (mp4v, avc1, XVID, MJPG) in order of preference
   - If you encounter video saving issues, install additional codecs:
     ```bash
     sudo apt-get install ubuntu-restricted-extras
     ```

2. **OpenCV Headless Mode**
   - For servers without GUI, use opencv-python-headless:
     ```bash
     pip uninstall opencv-python
     pip install opencv-python-headless
     ```

3. **File Permissions**
   - Ensure write permissions in the execution directory:
     ```bash
     chmod -R 755 ./
     ```
   - For storage directory:
     ```bash
     chmod -R 777 predictions_storage/
     ```

4. **CUDA Support**
   - Verify CUDA installation:
     ```python
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - Install CUDA toolkit if needed: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Troubleshooting

If you encounter:
- **Video writer initialization failed**: Try installing additional codecs or use the MJPG fallback
- **Permission denied errors**: Check directory permissions and ownership
- **OpenCV import errors**: Install system dependencies mentioned above
- **CUDA not found**: Verify CUDA installation and PyTorch CUDA support

## Citation

If you use this project, please cite:

```bibtex
@article{wang2017predrnn,
  title={PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs},
  author={Wang, Yunbo and Long, Mingsheng and Wang, Jianmin and Gao, Zhifeng and Yu, Philip S},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  pages={879--888},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCF101 dataset creators
- Implementation inspired by various research papers in video prediction
- Contributors and maintainers of PyTorch and related libraries