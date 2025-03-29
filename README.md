# 📦 Amazon Sales Data: Machine Learning Analysis & Prediction

This project analyzes Amazon sales data using data preprocessing, visualization, and machine learning models. It includes both regression and classification tasks using Random Forest models to predict profits and sales channel types.

---

## 📁 Dataset

👉 [Download AmazonSalesData.csv](AmazonSalesData.csv)

---

## 📊 Project Features

- Clean & preprocess raw sales data
- Feature engineering (e.g., delivery time)
- Label encoding and feature scaling
- Exploratory Data Analysis (EDA) with seaborn/matplotlib
- Random Forest Regressor to predict `Total Profit`
- Random Forest Classifier to predict `Sales Channel` (Online/Offline)
- Model evaluation with metrics: MAE, RMSE, Accuracy, Classification Report

---

## 🛠️ Setup Instructions

### 1. Install Python 3.11.4 and Add to PATH

**Windows**:  
- Download from [python.org](https://www.python.org/)
- Check "Add Python to PATH" during installation  
- Verify installation:
```bash
python --version
```

**macOS/Linux**:
```bash
brew install python@3.11
# or
sudo apt update && sudo apt install python3.11
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/YourUsername/Amazon-Sales-ML.git
cd Amazon-Sales-ML
```

---

### 3. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### ▶️ Run the Project

```bash
python MachineLearning.py
```

---

## 📈 Visualizations

- Distribution of Total Profit
- Sales Channel Split (Online vs Offline)
- Profit by Item Type
- Revenue vs Cost Scatterplot
- Feature Correlation Heatmap

---

## 🤖 ML Models

#### Random Forest Regressor – `Total Profit`
- MAE, MSE, RMSE

#### Random Forest Classifier – `Sales Channel`
- Accuracy, Classification Report

---

## 📁 Structure

```
├── MachineLearning.py
├── AmazonSalesData.csv
├── README.md
├── requirements.txt
```

---

## 📬 Contact

For queries or collaboration, feel free to reach out!
