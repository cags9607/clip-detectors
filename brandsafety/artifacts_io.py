import joblib

def save(obj, path):
    joblib.dump(obj, path)

def load(path):
    return joblib.load(path)
