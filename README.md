# 🖼️ Image-Toolkit (Flask + OpenCV)

This is a simple **Flask project** made for learning and practicing **Image Processing** using **Python** and **OpenCV**.  
It allows you to upload an image, apply different effects, and download the processed image easily.

---

## 🎯 Project Overview

This web app is built using Flask as the backend and OpenCV for image processing.  
You can use this project to understand how image filters work and how APIs handle images in web applications.

---

## 🧩 Features

- Upload any image and apply effects:
  - Grayscale
  - Blur
  - Sketch
  - Enhance (Contrast)
  - Face Detection
  - Posterize (Color reduction)
- Download the processed image in PNG, JPG, or WEBP format.
- User-friendly interface (HTML + CSS + JS).

---

## 🛠️ Tools and Technologies

- **Python 3**
- **Flask**
- **OpenCV**
- **NumPy**
- **HTML / CSS / JavaScript**

---

## 📁 Folder Structure

```
project-folder/
│
├── app.py                      # Main Flask file
├── templates/
│   └── index.html              # Frontend HTML file
├── static/
│   ├── css/                    # CSS files
│   └── js/                     # JavaScript files
├── haarcascade_frontalface_default.xml   # For face detection
└── README.md                   # Project documentation
```

---

## ⚙️ How to Run the Project

### Step 1: Clone or Download the Project
```bash
git clone https://github.com/your-username/flask-image-processing.git
cd flask-image-processing
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate  # On Linux/Mac
```

### Step 3: Install Required Packages
```bash
pip install flask opencv-python numpy
```

### Step 4: Run the Flask App
```bash
python app.py
```

Then open your browser and go to:  
👉 http://127.0.0.1:5000

---

## 🧠 API Routes

### `/api/process`
- Accepts an image and a selected tool name.
- Returns the processed image in PNG format.

### `/api/download`
- Allows you to download an image in a selected format (png, jpg, webp).

---

## 🧪 Example Python Request

```python
import requests

url = "http://127.0.0.1:5000/api/process"
files = {'image': open('test.jpg', 'rb')}
data = {'tool': 'grayscale'}
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open('output.png', 'wb') as f:
        f.write(response.content)
```

---

## 🧑‍💻 Created By

**Name:** Uttam Singh  
**Course:** MCA (Data Science)  
**University:** Dev Bhoomi Uttarakhand University  

This project is for educational purposes to understand how image processing and Flask API work together.

---

## 📜 License

Free to use for learning and personal projects.
