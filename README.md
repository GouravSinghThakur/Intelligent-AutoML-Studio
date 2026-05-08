# 🤖 Intelligent AutoML Studio
**Your Personal AI Data Scientist**

Intelligent AutoML Studio is a smart, easy-to-use platform that automatically builds machine learning models for you. You don't need to be a programmer or a data scientist to use it! Just upload your data, and the app will clean it, find the best patterns, and give you predictions and beautiful charts.

---

## 🌟 What does it do?

Imagine you have an Excel spreadsheet of data. Maybe it's a list of houses with their prices, or a list of customers and whether they bought a product. 

Normally, predicting future prices or customer behavior requires writing complex code. **Intelligent AutoML Studio does all of that automatically.**

- **You give it data:** Upload a `.csv` or `.xlsx` file.
- **It cleans the data:** It automatically fixes missing information and formats everything perfectly.
- **It tests different "AI brains":** It tries 14 different AI algorithms behind the scenes to see which one is the smartest for your specific data.
- **It gives you answers:** It shows you easy-to-understand charts, lets you make new predictions, and even lets you download the winning AI model!

---

## ✨ Key Features (Simple terms)

| Feature | What it means for you |
|---------|-----------------------|
| **Auto-Detection** | The app automatically figures out if you are trying to guess a **Category** (like "Spam" vs "Not Spam") or a **Number** (like "House Price: $300,000"). |
| **Smart Data Cleaning** | Messy data? Missing values? The app cleans it up automatically so you don't get errors. |
| **14 AI Algorithms** | It acts like 14 different experts analyzing your data, and it automatically picks the one that is the most accurate. |
| **Instant Predictions** | Type in new information (like the square footage of a new house) and the app will instantly predict the result. |
| **Interactive Charts** | Beautiful, dark-mode graphs that show you exactly how well the AI is performing. |
| **Download the Brain** | Once the app finds the best model, you can download it with one click to use elsewhere! |

---

## 🚦 How to use it (4 Easy Steps)

1. **📤 Upload Data:** Go to the "Data Upload" page and drag & drop your Excel or CSV file.
2. **🎯 Train the AI:** Go to "Model Training", tell the app what column you want to predict (e.g., "Price"), and click "Auto Train". Grab a coffee while it works!
3. **📊 View Results:** Check the "Visualisation" page to see beautiful charts explaining the results.
4. **🔮 Predict:** Go to the "Prediction" page, enter new details, and see what the AI guesses!

---

## 🛠️ For the Tech-Savvy (Under the Hood)

If you are a developer or recruiter, here is what powers the engine:

### 🧠 Supported Algorithms (14 Total)
- **7 Classification Models:** Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost, K-Nearest Neighbours, Gradient Boosting, Extra Trees.
- **7 Regression Models:** Ridge Regression, SVR, Random Forest Regressor, XGBoost Regressor, KNN Regressor, Gradient Boosting Regressor, Extra Trees Regressor.

### ⚙️ Tech Stack
- **Frontend:** Streamlit (Python-based UI)
- **Machine Learning:** scikit-learn, XGBoost
- **Hyperparameter Tuning:** Optuna (Bayesian optimization)
- **Visualisations:** Plotly, Matplotlib, Seaborn

### 📂 Project Structure
This app follows a modern, modular architecture:
- `app.py`: The main entry point.
- `src/`: Core logic (data processing, model registries, training loops, metric calculations).
- `ui/`: User interface components (pages, styling, sidebar).

## 🚀 Installation & Setup

Want to run this on your own computer? 

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
