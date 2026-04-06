import joblib
import pandas as pd
import os
from datetime import datetime

def check_live_data(app_num, ric, fid, message_count, model_dir="models"):
    """
    Example function that another app would use to check if current activity is normal.
    """
    model_path = os.path.join(model_dir, "anomaly_model.joblib")
    if not os.path.exists(model_path):
        return "Model not found. Train first."

    # 1. Load Artifacts
    model = joblib.load(model_path)
    le_app = joblib.load(os.path.join(model_dir, "encoder_app_number.joblib"))
    le_ric = joblib.load(os.path.join(model_dir, "encoder_RIC.joblib"))
    le_fid = joblib.load(os.path.join(model_dir, "encoder_FID.joblib"))

    # 2. Preprocess current context
    now = datetime.now()
    
    try:
        data = {
            'app_number_enc': le_app.transform([app_num])[0],
            'RIC_enc': le_ric.transform([ric])[0],
            'FID_enc': le_fid.transform([fid])[0],
            'hour': now.hour,
            'minute': now.minute,
            'day_of_week': now.weekday(),
            'message_count': message_count
        }
    except ValueError as e:
        return f"Unknown App/RIC/FID identity: {e}"

    # 3. Predict
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0] # 1 = Normal, -1 = Anomaly
    
    return "NORMAL" if prediction == 1 else "ANOMALY DETECTED (Silent/Irregular)"

if __name__ == "__main__":
    # Simulate a check
    print(f"Checking App 101, TRI.N, LAST with 0 messages...")
    result = check_live_data(101, "TRI.N", "LAST", 0)
    print(f"Result: {result}")
