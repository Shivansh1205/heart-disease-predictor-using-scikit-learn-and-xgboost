# Heart Disease Detection using Decision Trees and XGBoost

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xJSWtuuIxKvq5-ZcQG7btZZC0L8tt6XU#scrollTo=qlOyZdxdSow7)

---

## 📌 Project Overview
This project applies **machine learning** to predict the likelihood of heart disease based on patient health attributes.  
Two models were implemented:  
- A **Decision Tree Classifier (Scikit-learn)** for interpretability  
- An **XGBoost Classifier** for higher accuracy  

The goal is to explore the trade-off between interpretability and predictive power in medical applications.

---

## 🧠 Key Concepts
- **Decision Trees (Scikit-learn)**: Simple, interpretable flowchart-based classifier.  
- **XGBoost**: Gradient boosting algorithm known for high accuracy and efficiency.  
- **Evaluation Metrics**: Accuracy, precision, recall, confusion matrix, feature importance.  

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
   - **Decision Tree Classifier (Scikit-learn)**  
   - **XGBoost Classifier** (optimized for performance)  

3. **Model Evaluation**  
   - Compare Decision Tree vs. XGBoost  
   - Accuracy scores and confusion matrices  
   - Feature importance visualization  

---

## 🚀 Running on Google Colab
1. Open the notebook by clicking the **Colab badge** above.  
2. Install dependencies (optional, most are pre-installed on Colab):  
   ```python
   !pip install -r requirements.txt
