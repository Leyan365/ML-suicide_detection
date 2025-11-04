# 🧠 Suicide Detection using LSTM + GloVe

This project builds a deep learning model that detects suicidal text posts using **pre-trained GloVe embeddings (840B, 300d)** and an **LSTM neural network**.

## 📊 Dataset
- Source: [Suicide_Detection.csv](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- Balanced dataset with 232,074 entries  
  - **116,037 suicide**
  - **116,037 non-suicide**

## 🧩 Key Steps
- Text cleaning using `neattext`
- Tokenization and padding with Keras
- GloVe embedding matrix creation (`840B.300d`)
- Label encoding for classification
- (Next Step) LSTM + Dense layers for final prediction


## 🧰 Tools Used
- Python, Pandas, Scikit-learn, TensorFlow/Keras
- NeatText for text cleaning
- GloVe embeddings for transfer learning
- Jupyter Notebook
- Git + GitHub for version control

## 🚀 Future Work
- Train LSTM model and evaluate metrics
- Integrate with **Streamlit** for live prediction
- Deploy on **Hugging Face Spaces** or **Streamlit Cloud**

---

**Author:** [Leyan365](https://github.com/Leyan365)  
🌐 GitHub Repository: [ML-suicide_detection](https://github.com/Leyan365/ML-suicide_detection)
