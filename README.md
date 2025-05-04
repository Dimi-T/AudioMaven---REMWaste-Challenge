# 📬 AudioMaven - REMWaste Challenge  
**AudioMaven** is an accent classifier developed for the **REMWaste Coding Challenge**. It allows users to analyze spoken English from online Youtube videos and classify the speaker's accent using a pre-trained model, also providing the training piepline. The model is build using a simple, yet powerful **Convolutional Recurrent Neural Network** architecture.
## 🖥️ Setup Instructions (Ubuntu 20.04)  
### 🔁 1. Clone the Repository  
```bash
git clone https://github.com/Dimi-T/AudioMaven---REMWaste-Challenge.git
cd AudioMaven---REMWaste-Challenge
```  
### ⚙️ 2. Setup the Environment  
Make the setup script executable and run it with superuser privileges:  
```bash
chmod +x setup.sh
sudo ./setup.sh
```  
### 🧠 3. Choose a Model  
You can:  
- ✅ Use the pretrained model provided in the `Detector/models/` directory  
- 🧪 Train your own model using:  
  - The Jupyter notebook provided, or  
  - The full training pipeline available at: [Training Pipeline](https://www.kaggle.com/code/dimitrietanasescu/audiomaven)  
### 🎯 4. Run the Detector  
Navigate to the detector's source directory and run the script with your desired video URL:  
```bash
cd Detector/src
python main.py --url="your video url here"
```  
This will:  
- Download the video  
- Extract and process the audio  
- Classify the speaker’s accent  
## 📁 Project Structure  
```
AudioMaven---REMWaste-Challenge/
├── Detector/
│   └── src/
├── Training/
├── models/
├── setup.sh
└── README.md
```  
## 🧰 Requirements  
- Ubuntu 20.04  
- Python 3.8+  
- ffmpeg  
- Python packages

 
© 2025 Dimitrie Tănăsescu
