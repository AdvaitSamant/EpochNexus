import os
import pickle
import pandas as pd

class AIModelManager:
    def __init__(self):
        self.model = None
        self.data = None
        self.dataframe = None

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                self.dataframe = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.dataframe = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Please use CSV or Excel files.")
            
            self.data = file_path
            print(f"Data loaded from {file_path}")
            print(f"Dataset shape: {self.dataframe.shape}")
            return True
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    def get_data_preview(self, rows=5):
        if self.dataframe is None:
            return None
        return self.dataframe.head(rows)

    def get_dataset_info(self):
        if self.dataframe is None:
            return None
        
        info = {
            'shape': self.dataframe.shape,
            'columns': list(self.dataframe.columns),
            'dtypes': self.dataframe.dtypes.to_dict(),
            'missing_values': self.dataframe.isnull().sum().to_dict()
        }
        return info

    def train_model(self):
        """Train the AI model using loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        # Placeholder: replace with actual training code
        self.model = "trained_model_placeholder"
        print("Model trained (placeholder)")

    def save_model(self, file_path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Call train_model() first.")
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load a previously trained model from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {file_path}")
