import pandas as pd
import numpy as np
import random
from datetime import timedelta, date

def generate_pcos_patient_data(patient_id, num_cycles=24):
    base_cycle_length = random.randint(35, 55)
    variability = random.randint(15, 30)
    
    patient_records = []
    current_date = date(2023, random.randint(1, 12), random.randint(1, 28))

    for i in range(num_cycles):
        stress_level = random.randint(1, 10)
        
        cycle_length_noise = random.uniform(-variability, variability)
        stress_effect = 0
        if stress_level > 7:
            stress_effect = random.randint(5, 20) 
        
        cycle_length = int(base_cycle_length + cycle_length_noise + stress_effect)
        cycle_length = max(21, min(200, cycle_length))

        record = {
            'patient_id': patient_id,
            'cycle_id': i + 1,
            'start_date': current_date.strftime('%Y-%m-%d'),
            'cycle_length': cycle_length,
            'stress_level_of_cycle': stress_level
        }
        patient_records.append(record)
        
        current_date += timedelta(days=cycle_length)
        
    return patient_records

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    
    all_patient_data = []
    for patient_id in range(1, 101): 
        all_patient_data.extend(generate_pcos_patient_data(patient_id))
    
    df = pd.DataFrame(all_patient_data)
    
    output_filename = 'synthetic_pcos_cycle_data.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"Dataset created: '{output_filename}'")
    print("\nData Sample:")
    print(df.head())