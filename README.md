# Automatic Modulation Classification using CNN-LSTM

A deep learning system that identifies different types of wireless signal modulations. This project was our minor project at the Department of Electronics and Computer Engineering, Thapathali Campus, IOE, Nepal.

## Team Members
- Krishna Acharya
- Dinanath Padhya
- Bipul Kumar Dahal

## What This Project Does

In wireless communication, signals are modulated (encoded) in different ways. This project uses deep learning to automatically recognize which modulation technique is being used in a signal - kind of like teaching a computer to "listen" and identify different types of radio signals.

We built a system that can identify 24 different modulation types from real radio signals, even when the signals are noisy. The system uses a combination of CNN (for recognizing patterns) and LSTM (for understanding sequences over time) neural networks.

## Key Features

### What Makes It Work
- Uses AlexNet (a CNN) to look at signal patterns
- LSTM network to understand how signals change over time
- Attention mechanism to focus on the important parts
- Trained on RadioML 2018.01A dataset with real radio signals

### What It Can Classify
The system can recognize 24 different types of modulation schemes:
- Digital modulations: BPSK, QPSK, 8PSK, 16QAM, 64QAM, and many more
- Analog modulations: AM, FM
- Other schemes: GMSK, OQPSK

### Training Features
- Saves progress automatically so you can stop and resume training
- Tracks accuracy and loss over time
- Works with different noise levels (SNR from -20 to +30 dB)
- Visual plots to see how well it's learning

## Prerequisites

### Software Requirements
- Python 3.8 or higher
- PyTorch 1.10+ with CUDA support (recommended)
- Jupyter Notebook or JupyterLab
- Git

### Hardware Requirements
- NVIDIA GPU with 8GB+ VRAM (recommended for training)
- 16GB+ RAM
- 50GB+ storage for datasets

### Python Libraries
```
torch
torchvision
numpy
## What You Need

### Software
- Python 3.8 or newer
- PyTorch (GPU version recommended for faster training)
- Jupyter Notebook
- Basic Python libraries: numpy, pandas, matplotlib, h5py
## How to Set It Up

1. **Clone this repository**:
```bash
git clone https://github.com/whoisdinanath/amc-using-cnn-lstm.git
cd amc-using-cnn-lstm
```

2. **Install the required libraries**:
```bash
pip install torch torchvision numpy pandas h5py matplotlib seaborn scikit-learn
```

3. **Get the dataset**:
   - Download RadioML 2018.01A from [DeepSig](https://www.deepsig.ai/datasets)
   - Put the file somewhere and update the path in `train.ipynb`

## How to Use

### Training the Model

1. Open `train.ipynb` in Jupyter:
```bash
jupyter notebook train.ipynb
```

2. Update the dataset path in the notebook to point to your downloaded dataset

3. Run the cells - the notebook will:
   - Load the radio signals
   - Train the neural network
   - Save checkpoints automatically
   - Show you graphs of how well it's learning

Training can take several hours depending on your hardware. The model saves its progress, so you can stop and resume anytime.

### Testing the Model

Open `inference.ipynb` to test the trained model on new data. You'll see:
- How accurate it is at different noise levels
- Which modulation types it struggles with
- Confusion matrix showing what it mistakes for what

### Playing with Parameters

Use `parameter_tuning.ipynb` if you want to experiment with different settings to improve accuracy.

### Visualizing Results

The `plots.ipynb` notebook lets you create nice visualizations of the signals and results.

## Project Files

```
├── train.ipynb                  # Main training notebook
├── inference.ipynb              # Testing the model
├── parameter_tuning.ipynb       # Experimenting with settings
├── plots.ipynb                  # Making graphs and visualizations
├── test_radioml_dataset.py      # Check if dataset is loaded correctly
└── checkpoints/                 # Saved models
```

## Training Configuration

Default hyperparameters:
```python
params = {
    "num_classes": 24,
    "num_seq": 8,              # Sequence length
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 10,
    "dropout": 0.5,
    "gradient_clip": 1.0
}
```
## How It Works (Technical Details)

The model works in three stages:

1. **CNN (AlexNet)**: Takes the radio signal and extracts important features, similar to how you'd recognize patterns in an image

2. **LSTM**: Looks at sequences of these features over time to understand temporal patterns

3. **Attention + Classifier**: Focuses on the most important parts and makes the final decision about which modulation type it is

Default settings we used:
## Challenges We Faced

**Dataset Issues**: We started with a smaller custom dataset but later switched to RadioML's 24-class dataset. Had to rewrite a lot of code to handle the different format.

**Memory Problems**: The full dataset is huge and our computers kept running out of memory. We solved this by loading data in smaller batches.

**Training Stability**: The model was hard to train at first - it would sometimes explode or get stuck. We fixed this with gradient clipping and better learning rates.

**Capturing Time Patterns**: Radio signals have patterns over time that are tricky to learn. Adding LSTM and attention mechanism really helped with this.
- Around 85-90% accuracy overall
- Over 95% accurate when the signal is clean (high SNR)
- Still works at about 60% even with noisy signals
- Can classify signals in real-time

Some modulation types are easier to identify than others. Check out the inference notebook for detailed results.

## Desktop Application

We've also built a PyQt desktop application for this project! You can interact with the model through a user-friendly GUI interface.

**Check it out here**: [Automatic RF Identification Desktop App](https://github.com/krishna-ji/automatic-rf-identification-for-intelligent-communication)

## License

MIT License - feel free to use this code for your own projects!

## Acknowledgments

Big thanks to:
- DeepSig for the RadioML dataset
- Our professors and advisors at Thapathali Campus for their guidance
- The PyTorch team and open-source community
- Everyone who's contributed to research on deep learning for radio signals

## References

- RadioML 2018.01A Dataset: https://www.deepsig.ai/datasets
- O'Shea, T. J., et al. "Over-the-Air Deep Learning Based Radio Signal Classification"
- PyTorch: https://pytorch.org/

## Questions?

Feel free to open an issue on GitHub if you have questions or run into problems.

---

Made by Krishna Acharya, Dinanath Padhya, and Bipul Kumar Dahal  
Electronics and Computer Engineering, Thapathali Campus, IOE, Nepal
