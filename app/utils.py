# app/utils.py

import joblib
import os
from src import config

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    # Add other libraries' seed settings if used

def save_model(model, model_name):
    """
    Saves the trained model to disk.
    """
    model_dir = os.path.join(config.MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_name):
    """
    Loads a trained model from disk.
    """
    model_path = os.path.join(config.MODELS_DIR, model_name, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
