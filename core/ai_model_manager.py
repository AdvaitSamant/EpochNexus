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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve

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

    def train_model(self, target_column, model_type="LogisticRegression", test_size=0.2, random_state=42, **kwargs):
        
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
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

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
    
    def plot_confusion_matric(self, save_path = "confusion_matrix.png"):
        if self.model is None:
            raise ValueError("No model trained. Please train a model before plotting.")
        
        preds = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm,annot = True, fmt = "d", cmap = "Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def plot_roc_curve(self, save_path="roc_curve.png"):
        if self.model is None:
            raise ValueError("No trained model available.")
        if not hasattr(self.model, "predict_proba"):
            print("ROC curve requires probability estimates. Try SVM(probability=True) or tree/ensemble models.")
            return
        preds_proba = self.model.predict_proba(self.X_test)
        if preds_proba.shape[1] != 2:
            print("ROC Curve only available for binary classification.")
            return
        fpr, tpr, _ = roc_curve(self.y_test, preds_proba[:,1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
        plt.plot([0,1],[0,1], color="gray", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
        print(f"ROC curve saved to {save_path}")

    def plot_learning_curve(self, save_path="learning_curve.png"):
        if self.model is None:
            raise ValueError("No trained model available.")
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.figure()
        plt.plot(train_sizes, train_mean, "o-", label="Training score")
        plt.plot(train_sizes, test_mean, "o-", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend(loc="best")
        plt.savefig(save_path)
        plt.close()
        print(f"Learning curve saved to {save_path}")
