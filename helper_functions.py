from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd

def extract_all_features_to_csv(xml_file_path, output_csv_path=None):
    """
    Extract requested features from XML and create a CSV with one row per timestamp.
    
    Features extracted:
    - patient, timestamp, glucose_level, finger_stick, basal,
      meal_type, meal_carbs, exercise_intensity, exercise_duration,
      basis_heart_rate, basis_gsr, basis_skin_temperature,
      basis_air_temperature, basis_steps
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    patient_id = root.get('id', '')
    
    # Dictionary to store all data by timestamp
    # Key: timestamp string, Value: dict with all feature values
    data_by_ts = {}
    
    # Helper function to get or create entry for a timestamp
    def get_entry(ts_str):
        if ts_str not in data_by_ts:
            data_by_ts[ts_str] = {
                'patient': patient_id,
                'glucose_level': None,
                'finger_stick': None,
                'basal': None,
                'meal_type': None,
                'meal_carbs': None,
                'exercise_intensity': None,
                'exercise_duration': None,
                'basis_heart_rate': None,
                'basis_gsr': None,
                'basis_skin_temperature': None,
                'basis_air_temperature': None,
                'basis_steps': None
            }
        return data_by_ts[ts_str]
    
    # 1. Glucose Level
    for event in root.findall('glucose_level/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['glucose_level'] = value
    
    # 2. Finger Stick
    for event in root.findall('finger_stick/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['finger_stick'] = value
    
    # 3. Basal
    for event in root.findall('basal/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['basal'] = value
    
    # 4. Meal (extract both type and carbs)
    for event in root.findall('meal/event'):
        ts = event.get('ts')
        meal_type = event.get('type')
        carbs = event.get('carbs')
        if ts:
            entry = get_entry(ts)
            entry['meal_type'] = meal_type
            entry['meal_carbs'] = carbs
    
    # 5. Exercise (extract both intensity and duration)
    for event in root.findall('exercise/event'):
        ts = event.get('ts')
        intensity = event.get('intensity')
        duration = event.get('duration')
        if ts:
            entry = get_entry(ts)
            entry['exercise_intensity'] = intensity
            entry['exercise_duration'] = duration
    
    # 6. Basis Heart Rate
    for event in root.findall('basis_heart_rate/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['basis_heart_rate'] = value
    
    # 7. Basis GSR
    for event in root.findall('basis_gsr/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['basis_gsr'] = value
    
    # 8. Basis Skin Temperature
    for event in root.findall('basis_skin_temperature/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['basis_skin_temperature'] = value
    
    # 9. Basis Air Temperature
    for event in root.findall('basis_air_temperature/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['basis_air_temperature'] = value
    
    # 10. Basis Steps
    for event in root.findall('basis_steps/event'):
        ts = event.get('ts')
        value = event.get('value')
        if ts:
            entry = get_entry(ts)
            entry['basis_steps'] = value
    
    # Convert to DataFrame
    if not data_by_ts:
        print("No data found in XML file!")
        return None
    
    # Create list of records
    records = []
    for ts_str, values in data_by_ts.items():
        record = values.copy()
        record['timestamp'] = ts_str  # Add timestamp as a column
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Convert timestamp to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Reorder columns: timestamp first, then patient, then all other features
    feature_order = ['timestamp', 'patient', 'glucose_level', 'finger_stick', 'basal',
                     'meal_type', 'meal_carbs', 'exercise_intensity', 'exercise_duration',
                     'basis_heart_rate', 'basis_gsr', 'basis_skin_temperature',
                     'basis_air_temperature', 'basis_steps']
    
    # Ensure all columns exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = None
    
    df = df[feature_order]
    
    # Save to CSV
    if output_csv_path is None:
        output_csv_path = xml_file_path.replace('.xml', '_all_features.csv')
    
    df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Successfully extracted features!")
    print(f"   Input file: {xml_file_path}")
    print(f"   Output file: {output_csv_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {', '.join(feature_order)}")
    
    return df



def extract_duration_events_to_csv(xml_file_path, output_csv_path=None):
    """
    Extract events with ts_begin and ts_end from XML and create a CSV with duration.
    
    For each event with start and end times, extracts:
    - date (from ts_begin)
    - duration (calculated from ts_begin and ts_end in minutes)
    - sleep (quality)
    - basis_sleep (value)
    - work_intensity
    - bolus_type, bolus_dose, bolus_carb_input
    - temp_basal_value
    
    Returns a DataFrame with one row per event.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    patient_id = root.get('id', '')
    records = []
    
    # Helper function to calculate duration in minutes
    def calculate_duration(ts_begin_str, ts_end_str):
        """Calculate duration in minutes between two timestamps"""
        try:
            ts_begin = pd.to_datetime(ts_begin_str, dayfirst=True, errors='coerce')
            ts_end = pd.to_datetime(ts_end_str, dayfirst=True, errors='coerce')
            if pd.isna(ts_begin) or pd.isna(ts_end):
                return None
            duration = (ts_end - ts_begin).total_seconds() / 60  # Convert to minutes
            return duration
        except:
            return None
    
    # 1. Sleep events (has ts_begin, ts_end, quality)
    for event in root.findall('sleep/event'):
        ts_begin = event.get('ts_begin')
        ts_end = event.get('ts_end')
        quality = event.get('quality')
        
        if ts_begin and ts_end:
            duration = calculate_duration(ts_end, ts_begin)
            date = pd.to_datetime(ts_end, dayfirst=True, errors='coerce')
            
            records.append({
                'patient': patient_id,
                'timestamp': date,
                'duration_minutes': duration,
                'sleep': quality,
                'basis_sleep': None,
                'work_intensity': None,
                'bolus_type': None,
                'bolus_dose': None,
                'bolus_carb_input': None,
                'temp_basal_value': None
            })
    
    # 2. Work events (has ts_begin, ts_end, intensity)
    for event in root.findall('work/event'):
        ts_begin = event.get('ts_begin')
        ts_end = event.get('ts_end')
        intensity = event.get('intensity')
        
        if ts_begin and ts_end:
            duration = calculate_duration(ts_begin, ts_end)
            date = pd.to_datetime(ts_begin, dayfirst=True, errors='coerce')
            
            records.append({
                'patient': patient_id,
                'timestamp': date,
                'duration_minutes': duration,
                'sleep': None,
                'basis_sleep': None,
                'work_intensity': intensity,
                'bolus_type': None,
                'bolus_dose': None,
                'bolus_carb_input': None,
                'temp_basal_value': None
            })
    
    # 3. Bolus events (has ts_begin, ts_end, type, dose, bwz_carb_input or carb_input)
    for event in root.findall('bolus/event'):
        ts_begin = event.get('ts_begin')
        ts_end = event.get('ts_end')
        bolus_type = event.get('type')
        dose = event.get('dose')
        # Try both possible attribute names for carb input
        carb_input = event.get('bwz_carb_input') or event.get('carb_input')
        
        if ts_begin:
            # If ts_end exists, use it; otherwise duration is 0 (instantaneous bolus)
            if ts_end:
                duration = calculate_duration(ts_begin, ts_end)
            else:
                duration = 0  # Instantaneous bolus
            date = pd.to_datetime(ts_begin, dayfirst=True, errors='coerce')
            
            records.append({
                'patient': patient_id,
                'timestamp': date,
                'duration_minutes': duration,
                'sleep': None,
                'basis_sleep': None,
                'work_intensity': None,
                'bolus_type': bolus_type,
                'bolus_dose': dose,
                'bolus_carb_input': carb_input,
                'temp_basal_value': None
            })
    
    # 4. Temp Basal events (may have ts_begin/ts_end or just ts)
    for event in root.findall('temp_basal/event'):
        ts_begin = event.get('ts_begin')
        ts_end = event.get('ts_end')
        ts = event.get('ts')  # Fallback if no ts_begin
        value = event.get('value')
        
        # Use ts_begin if available, otherwise use ts
        start_time = ts_begin or ts
        if start_time:
            if ts_end:
                duration = calculate_duration(start_time, ts_end)
            else:
                duration = 0  # No end time specified
            date = pd.to_datetime(start_time, dayfirst=True, errors='coerce')
            
            records.append({
                'patient': patient_id,
                'timestamp': date,
                'duration_minutes': duration,
                'sleep': None,
                'basis_sleep': None,
                'work_intensity': None,
                'bolus_type': None,
                'bolus_dose': None,
                'bolus_carb_input': None,
                'temp_basal_value': value
            })
    
    # 5. Basis Sleep events (check if they have ts_begin/ts_end, otherwise use ts)
    for event in root.findall('basis_sleep/event'):
        ts_begin = event.get('ts_begin')
        ts_end = event.get('ts_end')
        ts = event.get('ts')
        value = event.get('value')
        
        # If it has ts_begin and ts_end, treat as duration event
        if ts_begin and ts_end:
            duration = calculate_duration(ts_begin, ts_end)
            date = pd.to_datetime(ts_begin, dayfirst=True, errors='coerce')
            
            records.append({
                'patient': patient_id,
                'timestamp': date,
                'duration_minutes': duration,
                'sleep': None,
                'basis_sleep': value,
                'work_intensity': None,
                'bolus_type': None,
                'bolus_dose': None,
                'bolus_carb_input': None,
                'temp_basal_value': None
            })
        # If only ts, create entry with 0 duration (point event)
        elif ts:
            date = pd.to_datetime(ts, dayfirst=True, errors='coerce')
            records.append({
                'patient': patient_id,
                'timestamp': date,
                'duration_minutes': 0,
                'sleep': None,
                'basis_sleep': value,
                'work_intensity': None,
                'bolus_type': None,
                'bolus_dose': None,
                'bolus_carb_input': None,
                'temp_basal_value': None
            })
    
    # Convert to DataFrame
    if not records:
        print("No duration events found in XML file!")
        return None
    
    df = pd.DataFrame(records)
    
    # Sort by date
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Reorder columns
    column_order = ['patient', 'timestamp', 'duration_minutes', 'sleep', 'basis_sleep',
                    'work_intensity', 'bolus_type', 'bolus_dose', 'bolus_carb_input',
                    'temp_basal_value']
    
    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    df = df[column_order]
    
    # Save to CSV
    if output_csv_path is None:
        output_csv_path = xml_file_path.replace('.xml', '_duration_events.csv')
    
    df.to_csv(output_csv_path, index=False)
    
    return df

# Example usage:
# df = extract_all_features_to_csv('559-ws-testing.xml')
# print(df.head())
# 
# df_duration = extract_duration_events_to_csv('559-ws-testing.xml')
# print(df_duration.head())