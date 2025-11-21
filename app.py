# app.py

import pandas as pd
import pickle

# Load model and scaler
try:
    with open("house_price_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

# Load dataset to get localities
df = pd.read_csv("CSV PATH HERE THE PATH OF kolkata_house_prices.csv")
localities = df['locality'].unique()
localities = sorted(localities)  # sort alphabetically
print(f"Loaded {len(localities)} localities from dataset.")

# One-hot encoding columns from training
X_train_columns = pd.get_dummies(df[['locality', 'area_sqft', 'bedrooms', 'bathrooms', 'floor', 'is_furnished']],
                                 columns=['locality'], drop_first=True).columns

def get_user_input():
    print("\nEnter the details to predict house price:")

    # Display localities with numbers
    print("\nSelect Locality:")
    for i, loc in enumerate(localities, start=1):
        print(f"{i}. {loc}")
    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(localities)}): "))
            if 1 <= choice <= len(localities):
                locality = localities[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(localities)}.")
        except ValueError:
            print("Invalid input. Enter a number.")

    area_sqft = float(input("Area in sqft: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    floor = input("Floor (number or Ground/Basement/5+ etc.): ").strip()
    furnished = input("Furnished? (Yes/No): ").strip().lower()

    # Process floor
    if floor.lower() in ['ground', 'g', 'gr']:
        floor = 0
    elif floor.lower() in ['basement', 'b', 'lower']:
        floor = -1
    elif '+' in floor:
        floor = int(floor.replace('+',''))
    else:
        try:
            floor = int(floor)
        except:
            floor = 0

    # Furnished
    is_furnished = 1 if furnished in ['yes','y','true','1'] else 0

    # Create DataFrame
    input_dict = {
        'area_sqft': [area_sqft],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'floor': [floor],
        'is_furnished': [is_furnished]
    }

    # One-hot encode locality
    for loc in localities:
        col_name = f"locality_{loc}"
        input_dict[col_name] = [1 if loc == locality else 0]

    # Create dataframe and ensure all columns match training
    input_df = pd.DataFrame(input_dict)
    for col in X_train_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X_train_columns]  # ensure correct order
    return input_df

while True:
    input_df = get_user_input()
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    print(f"\nPredicted Price: {prediction[0]:.2f} lakh\n")

    again = input("Do you want to predict another house price? (yes/no): ").strip().lower()
    if again not in ['yes','y']:
        break
