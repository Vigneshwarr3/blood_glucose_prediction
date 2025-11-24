"""
Simple Synthetic Data Generator for Blood Glucose Time Series Data
Uses statistical methods instead of GAN to generate synthetic patient data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def analyze_data_statistics(df):
    """
    Analyze statistical properties of the training data
    
    Returns:
        dict: Statistics for each numeric column
    """
    stats = {}
    
    for col in df.columns:
        if col in ['timestamp', 'patient']:
            continue
            
        # Get non-null values
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(values) > 0:
            stats[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'non_null_count': len(values),
                'null_rate': 1 - (len(values) / len(df)),
                'values': values.tolist() if len(values) < 1000 else None  # Store values for small sets
            }
        else:
            stats[col] = {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'median': 0,
                'non_null_count': 0,
                'null_rate': 1.0,
                'values': None
            }
    
    return stats


def generate_synthetic_data(df_train, n_samples=1000, start_date=None, patient_id=999):
    """
    Generate synthetic time series data based on training data statistics
    
    Args:
        df_train: Training DataFrame
        n_samples: Number of samples to generate
        start_date: Start date for synthetic data (default: current date)
        patient_id: Patient ID for synthetic data
    
    Returns:
        DataFrame with synthetic data
    """
    print("=" * 80)
    print("SIMPLE SYNTHETIC DATA GENERATION")
    print("=" * 80)
    
    # Analyze training data
    print("\n[1/4] Analyzing training data statistics...")
    stats = analyze_data_statistics(df_train)
    
    # Print summary
    print(f"   Analyzed {len(stats)} features")
    for col, stat in stats.items():
        if stat['non_null_count'] > 0:
            print(f"   {col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, "
                  f"range=[{stat['min']:.2f}, {stat['max']:.2f}], "
                  f"null_rate={stat['null_rate']:.2%}")
    
    # Set start date
    if start_date is None:
        start_date = datetime(2023, 1, 1, 0, 0, 0)
    elif isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Generate timestamps (assuming 1-minute intervals, similar to training data)
    print(f"\n[2/4] Generating {n_samples} synthetic samples...")
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='1min')
    
    # Initialize synthetic DataFrame
    synthetic_data = {
        'timestamp': timestamps,
        'patient': [patient_id] * n_samples
    }
    
    # Generate data for each feature
    print("\n[3/4] Generating feature values...")
    
    for col in df_train.columns:
        if col in ['timestamp', 'patient']:
            continue
        
        stat = stats[col]
        synthetic_values = []
        
        if stat['non_null_count'] == 0:
            # All null in training data - keep all null
            synthetic_values = [None] * n_samples
        else:
            # Generate values based on statistics
            null_rate = stat['null_rate']
            n_non_null = int(n_samples * (1 - null_rate))
            n_null = n_samples - n_non_null
            
            if stat['values'] is not None and len(stat['values']) < 100:
                # For small value sets, sample directly with replacement
                non_null_values = np.random.choice(stat['values'], size=n_non_null, replace=True)
            else:
                # Use normal distribution (clipped to min/max)
                mean = stat['mean']
                std = stat['std']
                
                # Generate values
                non_null_values = np.random.normal(mean, std, n_non_null)
                
                # Clip to reasonable bounds (within 3 std or min/max)
                lower_bound = max(stat['min'], mean - 3 * std)
                upper_bound = min(stat['max'], mean + 3 * std)
                non_null_values = np.clip(non_null_values, lower_bound, upper_bound)
            
            # Add temporal patterns for time-sensitive features
            if col in ['glucose_level', 'basis_heart_rate']:
                # Add daily cycle (sinusoidal pattern)
                hours = np.array([ts.hour + ts.minute/60 for ts in timestamps[:n_non_null]])
                daily_cycle = np.sin(2 * np.pi * hours / 24) * (stat['std'] * 0.3)
                non_null_values = non_null_values + daily_cycle
            
            # Add some temporal correlation (random walk component)
            if col in ['glucose_level']:
                # Add small random walk for glucose
                random_walk = np.cumsum(np.random.normal(0, stat['std'] * 0.1, n_non_null))
                non_null_values = non_null_values + random_walk
                # Re-clip after adding patterns
                non_null_values = np.clip(non_null_values, lower_bound, upper_bound)
            
            # Round to reasonable precision
            if col in ['glucose_level', 'finger_stick', 'basal', 'basis_heart_rate', 
                      'basis_skin_temperature', 'basis_air_temperature', 'basis_steps']:
                non_null_values = np.round(non_null_values, 1)
            elif col in ['basis_gsr']:
                non_null_values = np.round(non_null_values, 6)
            else:
                non_null_values = np.round(non_null_values, 2)
            
            # Create array with nulls randomly distributed
            all_values = list(non_null_values) + [None] * n_null
            np.random.shuffle(all_values)
            synthetic_values = all_values
        
        synthetic_data[col] = synthetic_values
    
    # Create DataFrame
    df_synthetic = pd.DataFrame(synthetic_data)
    
    # Ensure column order matches training data
    df_synthetic = df_synthetic[df_train.columns]
    
    print(f"\n[4/4] Generated {len(df_synthetic)} synthetic samples")
    print(f"   Date range: {df_synthetic['timestamp'].min()} to {df_synthetic['timestamp'].max()}")
    print(f"   Patient ID: {patient_id}")
    
    return df_synthetic


def add_temporal_patterns(df_synthetic, stats):
    """
    Add more realistic temporal patterns to synthetic data
    (e.g., meals cause glucose spikes, exercise affects heart rate)
    """
    # Add meal effects on glucose
    if 'glucose_level' in df_synthetic.columns and 'meal_carbs' in df_synthetic.columns:
        for idx in range(len(df_synthetic)):
            if pd.notna(df_synthetic.loc[idx, 'meal_carbs']) and df_synthetic.loc[idx, 'meal_carbs'] > 0:
                # Glucose spike after meal (peaks around 30-60 minutes later)
                spike_idx = min(idx + np.random.randint(30, 60), len(df_synthetic) - 1)
                if pd.isna(df_synthetic.loc[spike_idx, 'glucose_level']):
                    continue
                spike_amount = df_synthetic.loc[idx, 'meal_carbs'] * 2  # Rough estimate
                df_synthetic.loc[spike_idx, 'glucose_level'] = min(
                    df_synthetic.loc[spike_idx, 'glucose_level'] + spike_amount,
                    stats['glucose_level']['max']
                )
    
    # Add exercise effects
    if 'exercise_intensity' in df_synthetic.columns and 'basis_heart_rate' in df_synthetic.columns:
        for idx in range(len(df_synthetic)):
            if pd.notna(df_synthetic.loc[idx, 'exercise_intensity']) and df_synthetic.loc[idx, 'exercise_intensity'] > 0:
                # Heart rate increase during exercise
                if pd.isna(df_synthetic.loc[idx, 'basis_heart_rate']):
                    continue
                hr_increase = df_synthetic.loc[idx, 'exercise_intensity'] * 20
                df_synthetic.loc[idx, 'basis_heart_rate'] = min(
                    df_synthetic.loc[idx, 'basis_heart_rate'] + hr_increase,
                    stats['basis_heart_rate']['max']
                )
    
    return df_synthetic


def main():
    """Main function to generate synthetic data"""
    
    # Load training data
    print("Loading training data...")
    try:
        df_train = pd.read_csv('training_data.csv')
        df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
        print(f"✅ Loaded {len(df_train)} training samples")
    except FileNotFoundError:
        print("❌ Error: 'training_data.csv' not found!")
        print("   Please ensure the file exists in the current directory.")
        return
    
    # Generate synthetic data
    df_synthetic = generate_synthetic_data(
        df_train,
        n_samples=2000,  # Generate 2000 samples (~33 hours of 1-min intervals)
        start_date='2023-01-01 00:00:00',
        patient_id=999
    )
    
    # Add temporal patterns
    print("\nAdding temporal patterns (meal effects, exercise effects)...")
    stats = analyze_data_statistics(df_train)
    df_synthetic = add_temporal_patterns(df_synthetic, stats)
    
    # Save synthetic data
    output_file = 'synthetic_data_simple.csv'
    df_synthetic.to_csv(output_file, index=False)
    print(f"\n✅ Synthetic data saved to '{output_file}'")
    print(f"   Total rows: {len(df_synthetic)}")
    print(f"   Columns: {', '.join(df_synthetic.columns)}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print("\nTraining Data:")
    for col in ['glucose_level', 'basis_heart_rate', 'basis_steps']:
        if col in df_train.columns:
            values = pd.to_numeric(df_train[col], errors='coerce').dropna()
            if len(values) > 0:
                print(f"  {col}: mean={values.mean():.2f}, std={values.std():.2f}, "
                      f"min={values.min():.2f}, max={values.max():.2f}")
    
    print("\nSynthetic Data:")
    for col in ['glucose_level', 'basis_heart_rate', 'basis_steps']:
        if col in df_synthetic.columns:
            values = pd.to_numeric(df_synthetic[col], errors='coerce').dropna()
            if len(values) > 0:
                print(f"  {col}: mean={values.mean():.2f}, std={values.std():.2f}, "
                      f"min={values.min():.2f}, max={values.max():.2f}")
    
    print("\n" + "=" * 80)
    print("✅ Synthetic data generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

