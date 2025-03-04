# Audio MNIST Digit Recognition

Built an audio classification system that recognizes spoken digits (0–9) from raw audio files using Mel-Frequency Cepstral Coefficients (MFCC) and an Artificial Neural Network (ANN). This project demonstrates the transformation of raw audio into structured features suitable for accurate digit classification.

## Project Overview

This project applies deep learning techniques to the **Audio MNIST dataset**, which consists of spoken digits recorded by multiple speakers. By converting raw audio into MFCC spectrograms, we trained an ANN to classify digits with high accuracy, supporting real-world applications like voice-controlled systems and speech digitization.

## Dataset

- **Audio MNIST Dataset** containing `.wav` files of digits (0–9) spoken by different speakers.
- Each digit is stored as an individual audio file.
- Dataset structure:
  - 10 directories (digits 0–9).
  - Multiple `.wav` recordings for each digit.

## Objectives

- Preprocess and convert raw audio data into MFCC spectrograms.
- Visualize audio signals and their corresponding MFCCs.
- Train an Artificial Neural Network (ANN) to classify digits based on extracted features.
- Evaluate model performance and analyze misclassifications.
- Suggest future improvements for robustness and scalability.

## Methods

- **Audio Preprocessing**:
  - Used `librosa` for loading audio data and extracting MFCC features.
  - Generated 40-dimensional MFCC feature vectors for each audio file.
- **Feature Engineering**:
  - Converted time-series audio data into structured feature vectors.
  - Visualized waveforms and spectrograms for exploratory analysis.
- **Model Development**:
  - Built a fully connected ANN with:
    - Three hidden layers of 100 neurons each.
    - ReLU activations.
    - Softmax output layer for multi-class prediction.
  - Compiled with `Adam` optimizer and `sparse_categorical_crossentropy` loss.
- **Training and Evaluation**:
  - Trained the model for 100 epochs with a batch size of 32.
  - Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
  - Visualized model performance to identify areas of improvement.

## Results

- Achieved **99% accuracy** on the test set.
- High recall, precision, and F1-scores (**>98%**) across all digit classes.
- Minimal confusion between digits, with rare misclassifications (e.g., digit `0` occasionally predicted as `2`).
- MFCC spectrograms proved robust to variations in speaker tone, pitch, and accent.
- ANN architecture successfully generalized across diverse audio inputs without the need for convolutional layers due to the pre-processed spectrogram features.

## Business/Scientific Impact

- Demonstrates the ability to process and classify raw audio data effectively.
- Supports applications in:
  - Voice-activated systems.
  - Automated transcription services.
  - Speech recognition in low-resource environments.
- Provides a foundation for scalable speech digit recognition systems.

## Technologies Used

- Python
- Librosa
- TensorFlow (Keras)
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy
- Pandas

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/audio-mnist-digit-recognition.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Preprocess the audio data.
   - Extract MFCC features.
   - Train the ANN model.
   - Evaluate the model's performance.

## Future Work

- Implement **Convolutional Neural Networks (CNNs)** to capture spatial hierarchies in spectrogram data.
- Apply **data augmentation** to increase robustness against noise and speaker variability.
- Explore **transfer learning** from pre-trained audio models.
- Integrate real-time speech digit recognition for deployment in production environments.
