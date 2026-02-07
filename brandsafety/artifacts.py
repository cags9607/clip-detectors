import joblib

def save_artifact(obj, path):
    joblib.dump(obj, path)

def load_artifact(path):
    return joblib.load(path)
