import os 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class AIModelManager:
    def __init__(self):
        self.model = None
        self.data = None
        self.dataframe = None
        self.target_column = None
    
    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        try:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.csv':
                self.dataframe = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                self.dataframe = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
            
            self.data = file_path
            print(f"Data loaded from {file_path}")
            print(f"Dataset shape : {self.dataframe.shape}")
            return True
        
        except Exception as e:
            raise RuntimeError(f"Error loading file: {e}")
    
    def get_data_preview(self, rows = 5):
        if self.dataframe is None:
            return None
        return self.dataframe.head(rows)
    
    def get_dataset_info(self):
        if self.dataframe is None:
            return None
        info = {
            'shape' : self.dataframe.shape,
            'columns' : self.dataframe.columns.tolist(),
            'dtypes' : self.dataframe.dtypes.to_dict(),
            'missing_values' : self.dataframe.isnull().sum().to_dict()
        }
        return info

    def train_model(self, target_column, model_type= "Logistic Regression", text_size = 0.2, random_state = 42, **kwargs):
        """Train a model on the database with the chosen algorithm"""
        if self.dataframe is None: 
            raise ValueError("No data loaded. Please load data before training a model.")
        
        if target_column not in self.dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        self.target_column = target_column

        #Split the dataset 
        X = self.dataframe.drop(columns=[target_column])
        y = self.dataframe[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=text_size, random_state=random_state)   

        #Choose model
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(max_iter=500, **kwargs)
        elif model_type == "DecisionTree":
            self.model = DecisionTreeClassifier(**kwargs)
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == "KNN":
            self.model = KNeighborsClassifier(**kwargs)
        elif model_type == "SVM":
            self.model = SVC(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        #train the model
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        #Evaulation

        results = {
            "accuracy" : accuracy_score(y_test, preds),
            "report" : classification_report(y_test, preds, output_dict=True)
        }
        
        print(f"Model trained using {model_type}")
        print(f"Accuracy: {results['accuracy']}")
        return results
    
    def model_save(self, file_path):
        if self.model is None:
            raise ValueError("No model trained. Please train a model before saving.")
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {file_path}")
    
    def load_model (self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'rb') as f:
            self.model, self.target_column = pickle.load(f)
        print(f"Model loaded from {file_path}")