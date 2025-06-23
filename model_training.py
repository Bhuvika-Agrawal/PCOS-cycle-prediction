import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_model():
    print("--- Starting Model Training ---")

    try:
        df = pd.read_csv('synthetic_pcos_cycle_data.csv')
    except FileNotFoundError:
        print("Error: synthetic_pcos_cycle_data.csv not found.")
        print("Please run generate_data.py first.")
        return

    # Feature Engineering
    SEQUENCE_LENGTH = 3
    X, y = [], []

    for patient_id, group in df.groupby('patient_id'):
        cycles = group['cycle_length'].tolist()
        stress = group['stress_level_of_cycle'].tolist()

        if len(cycles) > SEQUENCE_LENGTH:
            for i in range(len(cycles) - SEQUENCE_LENGTH):
                feature_sequence = cycles[i:i + SEQUENCE_LENGTH]
                feature_sequence.extend(stress[i:i + SEQUENCE_LENGTH])
                X.append(feature_sequence)
                
                y.append(cycles[i + SEQUENCE_LENGTH])

    print(f"Created {len(X)} training sequences.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)
    
    print("Model training complete.")

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model evaluation on test data: Mean Absolute Error = {mae:.2f} days")

    # Save the trained model
    model_filename = 'pcos_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved successfully as '{model_filename}'")

if __name__ == "__main__":
    train_model()