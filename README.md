# Age and Gender Detection Projects

This repository contains three machine learning projects based on age and gender detection using computer vision and deep learning. These projects simulate real-world scenarios like mall security, amusement park ride checks, and fine-tuning pre-trained models for age estimation.

---

## 📁 Project List:

### 1️⃣ FineTune_AgeDetection
**Description:**  
Fine-tunes a pre-trained CNN model (like VGGFace) for accurate age prediction using datasets such as UTKFace (alternative to IMDB-WIKI due to size constraints). GUI is not required for this task.

📂 Folder: `FineTune_AgeDetection/`

---

### 2️⃣ HorrorRollerCoaster_Detection
**Description:**  
A real-time age and gender detection system designed for a horror roller coaster. The model flags people under 13 and over 60 as "Not Allowed" and marks them with a red rectangle in the video feed. It also logs age, gender, and entry time into an Excel or CSV file.

📂 Folder: `HorrorRollerCoaster_Detection/`

---

### 3️⃣ SeniorCitizen_Identification
**Description:**  
A real-time video detection system for malls or stores that identifies individuals over 60 as senior citizens, detects their gender, and logs their age, gender, and time of visit into a CSV or Excel file.

📂 Folder: `SeniorCitizen_Identification/`

---

## 💻 Installation

```bash
pip install -r requirements.txt
```txt
opencv-python
tensorflow
keras
numpy
pandas
h5py
