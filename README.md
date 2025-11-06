# 🧠 Suicide Detection using LSTM + GloVe

This project builds a deep learning model that detects suicidal text posts using **pre-trained GloVe embeddings (840B, 300d)** and an **LSTM neural network**.

## 📊 Dataset
- Source: [Suicide_Detection.csv](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- Balanced dataset with 232,074 entries  
  - **116,037 suicide**
  - **116,037 non-suicide**

The dataset is **balanced**, which helps avoid biased classification.

---

## 🧠 Model Architecture (LSTM + GloVe)

| Layer | Description |
|--------|-------------|
| **Embedding Layer** | GloVe `840B.300d` pretrained embeddings (non-trainable) |
| **LSTM (20 units)** | Extracts semantic meaning from text |
| **Global Max Pooling** | Reduces sequence output to key features |
| **Dense (256, relu)** | Fully connected layer |
| **Dense (1, sigmoid)** | Final binary classification (suicide / non-suicide) |

🟦 Total Parameters: **81,470,513**  
🟩 Trainable Params: **31,313** (99.96% frozen from GloVe)

---

## ✅ Model Performance

| Metric | Non-Suicide | Suicide |
|--------|-------------|----------|
| **Precision** | 0.90 | 0.96 |
| **Recall** | 0.96 | 0.90 |
| **F1-score** | 0.93 | 0.93 |

**Overall Test Accuracy:** ✅ **92.86%**


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

---

![Streamlit App Screenshot](assets/<img width="1340" height="863" alt="suicide ss" src="https://github.com/user-attachments/assets/be4fd77f-9afe-4549-99ec-fcff98e5ce02" />)


**Author:** [Leyan365](https://github.com/Leyan365)  
🌐 GitHub Repository: [ML-suicide_detection](https://github.com/Leyan365/ML-suicide_detection)
