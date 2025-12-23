# 🤟 ASL Sign Language Detector

Real-time American Sign Language (ASL) recognition with AI-powered translation and hands-free mouse control using computer vision and machine learning.



## 📊 Key Performance Metrics

- **98.97% Classification Accuracy** - Random Forest model trained on 2,600 hand gesture images
- **30 FPS Real-time Processing** - Smooth camera feed with 33ms frame processing time
- **21 Hand Landmarks Detection** - MediaPipe hand tracking with sub-pixel accuracy
- **40+ Language Translation** - Powered by Google Gemini 2.5 Flash API
- **5ms Mouse Response Time** - Ultra-low latency gesture-to-cursor control
- **26 ASL Letters** - Complete A-Z alphabet recognition

## ✨ Features

### 🔤 Detection Mode
- **Real-time ASL letter recognition** with confidence scoring
- **Auto-sentence builder** with smart spacing
- **AI translation** to 40+ languages (Google Gemini)
- **Text-to-speech** output with pyttsx3
- **Visual feedback** with confidence indicators

### 🖱️ Mouse Control Mode
- **Cursor control** - Point with index finger
- **Left click** - Pinch thumb + index finger
- **Right click** - Pinch thumb + middle finger
- **Drag & drop** - Open palm (4+ fingers extended)
- **Scroll** - Closed fist with vertical hand movement

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam
- 2GB free disk space (for training data)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/sign-language-detector-python.git
cd sign-language-detector-python
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download MediaPipe model:**
The `hand_landmarker.task` model is included in the repository (7.8 MB).

### Training Your Own Model

1. **Collect training images** (100 images per letter × 26 letters = 2,600 total):
```bash
python collect_imgs.py
```
- Press 'Q' when ready for each letter (A-Z)
- Hold the ASL sign steady while images are captured
- Images saved to `data/` directory

2. **Process images into dataset:**
```bash
python create_dataset.py
```
- Extracts 21 hand landmarks per image
- Creates `data.pickle` file (~1 MB)

3. **Train the classifier:**
```bash
python train_classifier.py
```
- Trains Random Forest model
- Outputs accuracy score
- Saves `model.p` file (~3 MB)

### Running the Application

```bash
streamlit run app_final.py
```

Open http://localhost:8501 in your browser.

## 📖 Usage Guide

### Detection Mode
1. Select **"Detection"** mode in the sidebar
2. Click **"START CAMERA"**
3. Show ASL letters with your **left hand** (for best results)
4. Letters are auto-detected and added to sentence
5. Use controls:
   - **SPACE** - Add space between words
   - **BACKSPACE** - Delete last character
   - **CLEAR** - Clear entire sentence
   - **TRANSLATE** - Translate to selected language
   - **SPEAK** - Text-to-speech output

### Mouse Control Mode
1. Select **"Mouse Control"** mode in the sidebar
2. Click **"START CAMERA"**
3. Use hand gestures:
   - ☝️ **Point** (index finger) - Move cursor
   - 👌 **Pinch** (thumb + index) - Left click
   - 🤏 **Pinch** (thumb + middle) - Right click
   - 🖐️ **Open palm** (4+ fingers) - Drag
   - ✊ **Closed fist** - Scroll up/down

### Settings (Sidebar)
- **Mode**: Switch between Detection and Mouse Control
- **Hold Time**: Adjust letter detection delay (0.5-3.0s)
- **Smoothing**: Mouse movement smoothing (1-10)
- **Target Language**: Choose translation language

## 🏗️ Project Structure

```
sign-language-detector-python/
├── app_final.py              # Main Streamlit application
├── collect_imgs.py           # Training data collection script
├── create_dataset.py         # Dataset preprocessing
├── train_classifier.py       # Model training script
├── translator.py             # Gemini API translation module
├── translation_config.py     # Translation settings
├── requirements.txt          # Python dependencies
├── hand_landmarker.task      # MediaPipe hand detection model (7.8 MB)
├── model.p                   # Trained Random Forest model (3 MB)
├── data.pickle               # Processed dataset (1 MB)
├── translation_cache.json    # Cached translations
├── data/                     # Training images (26 folders: 0-25 for A-Z)
│   ├── 0/                   # Letter 'A' images
│   ├── 1/                   # Letter 'B' images
│   └── ...
└── README.md                # This file
```

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core language | 3.8+ |
| **Streamlit** | Web UI framework | Latest |
| **OpenCV** | Computer vision & camera | 4.8+ |
| **MediaPipe** | Hand landmark detection | 0.10+ |
| **scikit-learn** | Random Forest classifier | 1.2+ |
| **NumPy** | Numerical computations | Latest |
| **Google Gemini API** | AI translation | 2.5 Flash |
| **pyttsx3** | Text-to-speech | 2.90+ |
| **autopy** | Mouse automation | Latest |

## 📈 Model Performance

### Random Forest Classifier
- **Training samples**: 2,600 images (100 per class)
- **Test accuracy**: 98.97%
- **Features**: 42 (21 landmarks × 2 coordinates)
- **Classes**: 26 (A-Z)
- **Training time**: ~3 seconds
- **Prediction time**: <5ms per frame

### Hand Detection
- **Model**: MediaPipe Hand Landmarker
- **Landmarks**: 21 points per hand
- **Detection confidence**: >90% in good lighting
- **Processing speed**: 30 FPS

## 🎯 Use Cases

- **Accessibility**: Enable communication for hearing/speech impaired individuals
- **Education**: Learn ASL alphabet interactively
- **Gaming**: Hands-free game control
- **Assistive Technology**: Control devices without physical contact
- **Sign Language Research**: Dataset collection and analysis

## ⚙️ Configuration

### Gemini API Setup (for translation)
1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Webcam Settings
- Default resolution: 1280×720 (720p)
- Frame rate: 30 FPS
- Camera index: 0 (change in code if needed)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check camera permissions, try different camera index |
| Low FPS | Close other apps using camera, reduce resolution |
| Poor detection accuracy | Ensure good lighting, use left hand, clear background |
| Translation not working | Check Gemini API key and quota limits |
| Mouse control lagging | Reduce smoothing value, check CPU usage |

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin master
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `app_final.py`
   - Add secrets in dashboard:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```

3. **Deploy!** Your app will be live in minutes.

📖 **Full deployment guide**: See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**⚠️ Note**: Mouse control features only work locally (requires system access). Detection mode works perfectly on cloud deployment.
