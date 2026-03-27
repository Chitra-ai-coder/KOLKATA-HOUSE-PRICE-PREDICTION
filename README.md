<div align="center">

# 🏡 Kolkata House Price Prediction

*A machine learning pipeline utilizing a Random Forest Regressor to predict real estate prices in Kolkata based on property attributes.*

</div>

---

## 📖 Overview

This project is a complete machine learning pipeline designed to predict house prices in Kolkata. It handles everything from raw data cleaning and preprocessing to model training, rigorous evaluation, and detailed visualization. It also features a user-friendly Command Line Interface (CLI) allowing users to input property details and get real-time price estimations.

## ✨ Key Features

* **Accurate Predictions:** Estimates house prices based on critical factors: locality, area (sq.ft.), bedrooms, bathrooms, floor level, and furnished status.
* **Robust Preprocessing:** Automatically handles numeric conversion, one-hot encoding for categorical variables, and feature scaling.
* **Performance Metrics:** Evaluates model accuracy using R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
* **Rich Visualizations:** Automatically generates insights including correlation heatmaps, price distributions, scatter plots, boxplots, pairplots, and feature importance charts.
* **Interactive CLI:** A seamless command-line tool for instantaneous, real-time predictions.

---

## 📂 Project Structure

```text
Kolkata_House_Price_Prediction/
│
├── kolkata_house_prices.csv   # Dataset containing Kolkata house details and prices
├── train_model.py             # Script to clean data, train the model, and generate visualizations
├── app.py                     # Interactive CLI script for making predictions
├── house_price_model.pkl      # Saved Random Forest model (generated post-training)
├── scaler.pkl                 # Saved StandardScaler (generated post-training)
└── README.md                  # Project documentation
```

---

## 📊 Dataset

The project utilizes `kolkata_house_prices.csv`, which contains historical data on real estate in Kolkata. This dataset serves as the foundation for training the Random Forest Regressor and generating exploratory data visualizations.

---

## ⚙️ Requirements & Installation

Ensure you have **Python 3.x** installed on your system. 

Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 Usage

### 1. Train the Model

Before making predictions, you need to train the model. Run the training script to process the data, train the algorithm, generate visual plots, and save the model artifacts (`house_price_model.pkl` and `scaler.pkl`).

```bash
python train_model.py
```

### 2. Predict Prices

Once the model is trained, launch the interactive CLI to get price estimates. You will be prompted to enter specific house details.

```bash
python app.py
```

**Example CLI Output:**
```text
Enter the locality: Salt Lake
Enter the area (sq. ft.): 1200
Enter number of bedrooms: 3
Enter number of bathrooms: 2
Enter the floor: 2
Is it furnished? (yes/no): yes

🏠 Predicted Price: 85.32 lakh
```
*(You can continue predicting multiple properties without restarting the app!)*

---

## 🧠 Why Random Forest?

We utilize a **Random Forest Regressor** for this task because it:
* Successfully captures complex, non-linear relationships in real estate data.
* Is highly robust to outliers and noise in the dataset.
* Significantly reduces the risk of overfitting compared to single decision trees.
* Provides clear insights into **feature importance** (e.g., determining if 'area' impacts price more than 'locality').

---

## 🛠️ Future Improvements

- [ ] **Graphical User Interface (GUI):** Build a desktop or web interface using Tkinter or Streamlit.
- [ ] **Feature Expansion:** Incorporate additional data points such as year built, distance to the nearest metro station, and local amenities.
- [ ] **Hyperparameter Tuning:** Implement GridSearch or RandomSearch to squeeze out even higher predictive accuracy.
- [ ] **API Deployment:** Wrap the model in a Flask or FastAPI backend to serve predictions over the web.
