import pickle
import numpy as np
import random

def predict_next_cycle(history_cycles, history_stress):
    """
    Predicts the next cycle length given historical data.
    """
    model_filename = 'pcos_model.pkl'
    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return "Error: Model file 'pcos_model.pkl' not found. Please run train_model.py first."

    if len(history_cycles) != 3 or len(history_stress) != 3:
        return "Error: Please provide a history of exactly 3 cycle lengths and 3 stress levels."
    
    # Combine the features in the same way we did for training
    features = history_cycles + history_stress
    
    # Reshape for the model and predict
    prediction = model.predict(np.array(features).reshape(1, -1))
    
    return int(prediction[0])

if __name__ == "__main__":
    # --- Example Usage ---
    # Here we simulate some sample historical data a user might provide.
    sample_cycle_history = [45, 62, 51]  # Last 3 cycle lengths in days
    sample_stress_history = [7, 4, 8]   # Corresponding stress levels (1-10)
    
    predicted_length = predict_next_cycle(sample_cycle_history, sample_stress_history)
    
    print("\n--- Prediction Example ---")
    print(f"Given cycle history (days): {sample_cycle_history}")
    print(f"Given stress history (1-10): {sample_stress_history}")
    print("-" * 26)
    print(f"Predicted length of the next cycle is: {predicted_length} days")