# 📰 Fake News Detection System

This project is a *Fake News Detection System* built using *Python, **Machine Learning* (Logistic Regression), *Natural Language Processing (NLP)* techniques, and *Streamlit* for the web interface.

## 📌 Objective
To detect whether a given news article is *Real* or *Fake* using Machine Learning.

## 📚 Technologies Used
- Python
- Pandas
- Scikit-learn (sklearn)
- Streamlit
- Joblib
- TF-IDF Vectorizer

## 🛠 Working
1. Dataset preparation (Fake.csv and True.csv)
2. Data cleaning and processing
3. Text vectorization using TF-IDF
4. Model training using Logistic Regression
5. Building a Streamlit Web Interface
6. Predicting Real or Fake news with confidence score

## 📂 Project Structure
- model.pkl - Trained Machine Learning Model
- vectorizer.pkl - Trained TF-IDF Vectorizer
- app.py - Web Interface using Streamlit
- Fake.csv, True.csv - Dataset Files

## 🚀 How to Run the Project
1. Clone the repository:
    bash
    git clone <repository-link>
    
2. Navigate to project folder:
    bash
    cd Fake-News-Detection-Project
    
3. Install dependencies:
    bash
    pip install -r requirements.txt
    
4. Run the app:
    bash
    streamlit run app.py
    

## 📈 Result
- Predicts whether the news is Real or Fake.
- Provides confidence percentage for prediction.

---
⭐️ Feel free to contribute and improve this project
