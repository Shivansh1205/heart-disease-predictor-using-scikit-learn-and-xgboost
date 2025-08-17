# Heart Disease Detection using Decision Trees

## 📌 Project Overview
This project implements a **Decision Tree classifier** to predict the likelihood of heart disease based on patient health attributes. The goal is to apply supervised machine learning techniques to medical data and build an interpretable model that can assist in identifying high-risk individuals.

---

## 🧠 Key Concepts
- **Decision Trees**: A flowchart-like structure that splits data into branches based on feature values, making classification results interpretable.
- **Supervised Learning**: The model is trained on labeled data (patients with and without heart disease).
- **Evaluation Metrics**: Accuracy, precision, recall, and confusion matrix are used to assess performance.

---

## 📂 Dataset
The dataset contains patient medical records with features such as:
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Serum Cholesterol  
- Fasting Blood Sugar  
- Resting ECG Results  
- Maximum Heart Rate Achieved  
- Exercise-Induced Angina  
- ST Depression Induced by Exercise  
- Slope of the Peak Exercise ST Segment  
- Number of Major Vessels Colored by Fluoroscopy  
- Thalassemia Result  

**Target variable**: Presence (1) or absence (0) of heart disease.

---

## ⚙️ Implementation Steps
1. **Data Preprocessing**  
   - Handle missing values  
   - Normalize/scale features (if required)  
   - Encode categorical variables  

2. **Model Training**  
   - Split dataset into **training** and **testing** sets  
   - Train a Decision Tree Classifier on training data  

3. **Model Evaluation**  
   - Evaluate accuracy on test set  
   - Generate confusion matrix  
   - Visualize decision tree for interpretability  

---

## 📊 Results
- **Accuracy**: ~ (varies depending on parameters, usually 70–80%)  
- **Confusion Matrix**: Shows true positives, false positives, etc.  
- **Tree Visualization**: Helps explain which health factors contribute most to predictions.  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-decision-tree.git
   cd heart-disease-decision-tree
