# Heart Disease Detection using Decision Trees

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xJSWtuuIxKvq5-ZcQG7btZZC0L8tt6XU#scrollTo=qlOyZdxdSow7)

---

## 📌 Project Overview
This project implements a **Decision Tree classifier** to predict the likelihood of heart disease using patient health attributes. The goal is to explore supervised machine learning in a healthcare setting and create an interpretable model that highlights important medical risk factors.

---

## 🧠 Key Concepts
- **Decision Trees**: Used for interpretable classification.  
- **Supervised Learning**: Model trained on labeled patient records.  
- **Evaluation Metrics**: Accuracy, precision, recall, and confusion matrix.  

---

## 📂 Dataset
The dataset contains medical records with features such as:
- Age  
- Sex  
- Chest Pain Type  
- Blood Pressure  
- Serum Cholesterol  
- Fasting Blood Sugar  
- Resting ECG Results  
- Maximum Heart Rate Achieved  
- Exercise-Induced Angina  
- ST Depression Induced by Exercise  
- Slope of ST Segment  
- Number of Major Vessels (Fluoroscopy)  
- Thalassemia Result  

**Target variable** → Presence (1) or Absence (0) of heart disease.

---

## ⚙️ Workflow
1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical variables  
   - Split into train/test sets  

2. **Model Training**  
   - Train a Decision Tree Classifier  
   - Visualize the decision tree  

3. **Evaluation**  
   - Accuracy score  
   - Confusion matrix  
   - Feature importance  

---

## 🚀 Running on Google Colab
1. Open the notebook by clicking the **Colab badge** above.  
2. Install dependencies (optional, most are pre-installed on Colab):  
   ```python
   !pip install -r requirements.txt
