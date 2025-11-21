A Python-based machine learning project to predict house prices in Kolkata using a Random Forest Regressor. It includes data cleaning, model training, evaluation, visualization, and a user-friendly CLI for predictions.
Features
Predict house prices based on locality, area, bedrooms, bathrooms, floor, and furnished status.
Preprocesses data with numeric conversion, one-hot encoding, and scaling.
Model evaluation using R² score, MAE, and RMSE.
Generates visualizations: correlation heatmap, price distribution, scatter plot, boxplot, pairplot, and feature importance.
Interactive CLI for real-time predictions.
Dataset
kolkata_house_prices.csv contains house details and prices in Kolkata.
Used for training the Random Forest Regressor and generating visualizations.
Project Structure
Kolkata_House_Price_Prediction/
│
├── kolkata_house_prices.csv
├── train_model.py       # Train model, visualize data
├── app.py               # CLI for predictions
├── house_price_model.pkl# Saved trained model (generated after training)
├── scaler.pkl           # Saved StandardScaler (generated after training)
├── README.md            # This file
Requirements
Python 3.x
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
Install via:
pip install pandas numpy scikit-learn matplotlib seaborn
Usage
Train the Model
python train_model.py
Cleans data, trains the Random Forest model, evaluates, and saves house_price_model.pkl & scaler.pkl.
Predict Prices
python app.py
Enter house details interactively.
Receive predicted price instantly.
Option to predict multiple houses.
Example Output
🏠 Predicted Price: 85.32 lakh
Why Random Forest?
Captures non-linear relationships, robust to outliers, reduces overfitting, and provides feature importance.
Future Improvements
Add GUI (Tkinter/Streamlit)
Include more features (year built, distance to metro, amenities)
Hyperparameter tuning for higher accuracy
Deploy as web API (Flask/FastAPI)
