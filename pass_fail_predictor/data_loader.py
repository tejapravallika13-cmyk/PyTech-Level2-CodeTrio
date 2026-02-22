import pandas as pd

def load_data(file_path):
    """
    Loads the student performance dataset from the given path.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None