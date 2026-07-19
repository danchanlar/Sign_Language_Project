# Greek Sign Language Public-Service Assistant

An AI-based Greek Sign Language recognition project designed for **public-service communication**.
The system processes Greek Sign Language sentence data, extracts hand and body landmarks with **MediaPipe Holistic**, trains sequence classification models, and provides predictions through a **Streamlit web application**.

> This project is a closed-domain assistant for predefined public-service sentences.
> It is not a general sign language translator and does not replace a professional interpreter.

---

## Project Overview

The goal of this project is to recognize predefined Greek Sign Language sentence instances related to public-service scenarios, such as:

* Citizen service centers
* Police / public safety services
* Municipal services
* Health services
* School offices

The model receives a short Greek Sign Language input and predicts the most likely **GSL gloss sentence**.
Where available, the app also displays a possible **natural Greek meaning** and a suggested response for the public-service employee.

---

## Dataset

The project uses a Greek Sign Language dataset containing continuous sentence instances.

The continuous part of the dataset is frame-based.
Each sentence is stored as a folder of `.jpg` frames instead of a single video file.

Example structure:

```text
GSL_continuous/
└── health1_signer2_rep1_sentences/
    └── sentences0004/
        ├── frame_0000.jpg
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        └── ...
```

The annotation files contain rows in the following format:

```text
video_key | gloss_sentence
```

Example:

```text
police1_signer3_rep1_sentences/sentences0000 | ΓΕΙΑ ΕΓΩ(1) ΜΠΟΡΩ ΒΟΗΘΩ
```

In this project:

* `video_key` is used to locate the corresponding frame folder.
* `gloss_sentence` is used as the prediction label.
* A manual mapping can optionally convert gloss sentences into more natural Greek text.

---

## Workflow

```text
Google Drive Dataset
│
├── GSL_continuous.tar.gz
│   └── frame folders with .jpg images
│
└── GSL_continuous_files/*.csv
    └── video_key | gloss_sentence

↓
Read annotation files
↓
Analyze dataset
↓
Select a small subset
↓
Extract only selected frame folders
↓
Sort frames temporally
↓
Create fixed-length sequences
↓
Extract MediaPipe Holistic landmarks
↓
Train GRU / LSTM / Transformer model
↓
Evaluate model
↓
Export artifacts
↓
Run Streamlit public-service assistant
```

---

## Methodology

### 1. Frame Extraction

The full dataset is large, so the project does not extract the entire archive.
Instead, it selects a smaller subset of sentence classes and extracts only the necessary frame folders.

Example subset:

```text
20 sentence classes × 10 samples per class = 200 samples
```

This makes the project suitable for Google Colab and avoids unnecessary storage usage.

---

### 2. Fixed-Length Sequence Creation

Each GSL sentence may contain a different number of frames.

To train a neural network, every sample is converted to the same sequence length:

```text
60 frames per sentence
```

If a sentence has more than 60 frames, frames are sampled uniformly.
If a sentence has fewer than 60 frames, the last frame is repeated until the sequence reaches 60 frames.

---

### 3. Landmark Extraction

MediaPipe Holistic is used to extract landmarks from each frame.

The feature vector contains:

```text
Left hand:  21 landmarks × 3 coordinates = 63 features
Right hand: 21 landmarks × 3 coordinates = 63 features
Pose:       33 landmarks × 4 values      = 132 features
```

Total:

```text
63 + 63 + 132 = 258 features per frame
```

Final input shape:

```text
60 frames × 258 features
```

---

### 4. Model Training

The project supports sequence classification models such as:

* GRU
* LSTM
* Transformer

The first working version uses a GRU-based model because it is lightweight and suitable for sequence data.

The model predicts one of the selected GSL gloss sentence classes.

---

## Streamlit Application

The project includes a Streamlit web app for public-service use.

The app allows the user to:

* Upload a short GSL video
* Run prediction
* View the predicted GSL gloss sentence
* View a possible natural Greek meaning
* See the confidence score
* View top alternative predictions
* Get a suggested public-service employee response
* Download the result as JSON

---

## Technologies Used

* Python
* Google Colab
* OpenCV
* MediaPipe Holistic
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Streamlit
* Cloudflare Tunnel

---

## Repository Structure

```text
.
├── GSL_Public_Service_Project_Corrected.ipynb
├── streamlit_app.py
├── requirements_colab.txt
├── README.md
├── workflow_diagram.py
└── outputs/
    ├── processed/
    ├── models/
    └── reports/
```

The dataset itself is not included in this repository because of its large size.

---

## Installation

This project was developed and tested in Google Colab.

Install the required packages:

```bash
pip install opencv-python mediapipe==0.10.21 tensorflow scikit-learn pandas numpy joblib matplotlib tqdm streamlit
```

For a more stable Colab setup, compatible versions can be installed manually:

```bash
pip install numpy==1.26.4 protobuf==4.25.9 tensorflow==2.19.1 keras==3.10.0 mediapipe==0.10.21 opencv-python==4.10.0.84
```

---

## Usage in Google Colab

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

The dataset is expected to be available at:

```text
/content/drive/MyDrive/GSL Dataset/
```

Expected files:

```text
GSL Dataset/
├── GSL_continuous.tar.gz
├── GSL_continuous_files/
├── GSL_isolated.tar.gz
└── GSL_iso_files/
```

---

### 2. Run the Notebook

Open and run:

```text
GSL_Public_Service_Project_Corrected.ipynb
```

The notebook performs:

1. Dataset loading
2. Dataset analysis
3. Subset selection
4. Frame extraction
5. Landmark extraction
6. Model training
7. Evaluation
8. Streamlit export

---

### 3. Run the Streamlit App

Start Streamlit:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

In Google Colab, use Cloudflare Tunnel to access the app:

```bash
./cloudflared tunnel --url http://localhost:8501 --protocol http2 --no-autoupdate
```

Then open the generated `trycloudflare.com` link.

---

## Example Input

Because the dataset is frame-based, an example `.mp4` video can be created from one extracted frame folder.

The Streamlit app can then be tested using that generated demo video.

Example expected output:

```text
Predicted GSL gloss sentence:
ΓΕΙΑ ΕΓΩ(1) ΜΠΟΡΩ ΒΟΗΘΩ

Possible natural Greek meaning:
Γεια σας, μπορώ να σας βοηθήσω.
```

---

## Evaluation

The notebook generates:

* Training accuracy curve
* Validation accuracy curve
* Training loss curve
* Validation loss curve
* Classification report
* Confusion matrix
* Error analysis file

These outputs are saved inside:

```text
GSL_public_service_outputs/reports/
```

---

## Limitations

This system has important limitations:

* It only recognizes the selected sentence classes used during training.
* It does not translate arbitrary Greek Sign Language.
* It depends on the quality of MediaPipe landmark detection.
* It may perform poorly on unseen signers, lighting conditions or camera angles.
* The natural Greek meaning is manually mapped and may not exist for all gloss sentences.
* It should not be used as a replacement for professional sign language interpretation.

---

## Future Improvements

Possible future extensions include:

* Training on more sentence classes
* Using signer-independent train/test split
* Adding support for uploaded `.zip` frame folders
* Improving the natural Greek translation mapping
* Adding text-to-speech output
* Adding webcam-based real-time recognition
* Fine-tuning a Transformer-based architecture
* Deploying the app permanently on Streamlit Community Cloud or another hosting service

---

## Project Status

Current implementation:

* Dataset annotations are loaded correctly
* Frame-based dataset structure is handled
* Selected frame folders are extracted successfully
* MediaPipe features are extracted from frame sequences
* A sequence classification model is trained
* A Streamlit demo app is available

---

## Disclaimer

This project is developed for educational and research purposes.
It is a closed-domain prototype for public-service communication and should not be considered a complete Greek Sign Language translation system.
