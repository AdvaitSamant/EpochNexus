import os 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import json

class AIModelManager:
    def __init__(self):
        self.model = None
        self.data = None
        self.dataframe = None
        self.target_column = None
        self.scaler = None
        self.training_history = {}
    
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
    
    def get_data_preview(self, rows=5):
        if self.dataframe is None:
            return None
        return self.dataframe.head(rows)
    
    def get_dataset_info(self):
        if self.dataframe is None:
            return None
        info = {
            'shape': self.dataframe.shape,
            'columns': self.dataframe.columns.tolist(),
            'dtypes': self.dataframe.dtypes.to_dict(),
            'missing_values': self.dataframe.isnull().sum().to_dict()
        }
        return info

    def _preprocess_data(self, X, y, scale=True):
        """Preprocess data: handle missing values and optional scaling"""
        X_processed = X.copy()
        
        # Handle missing values
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if X_processed[col].dtype in ['float64', 'int64']:
                    X_processed[col].fillna(X_processed[col].mean(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)
        
        # Handle categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_processed[col] = pd.factorize(X_processed[col])[0]
        
        # Scale features if needed
        if scale:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X_processed)
            else:
                X_processed = self.scaler.transform(X_processed)
        
        return X_processed

    def train_model_automatic(self, target_column, test_size=0.2, random_state=42):
        """Automatically select and train the best model with optimized settings"""
        if self.dataframe is None:
            raise ValueError("No data loaded. Please load data before training a model.")
        
        if target_column not in self.dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        self.target_column = target_column
        print(f"Starting automatic model training...")
        print(f"Target column: {target_column}")

        # Split the dataset
        X = self.dataframe.drop(columns=[target_column])
        y = self.dataframe[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Preprocess data
        X_train_processed = self._preprocess_data(X_train.copy(), y_train, scale=True)
        X_test_processed = self._preprocess_data(X_test.copy(), y_test, scale=True)

        # Define automatic model configurations
        models_config = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=500, random_state=random_state),
                'params': {}
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier(random_state=random_state),
                'params': {}
            },
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
                'params': {}
            },
            'KNN': {
                'model': KNeighborsClassifier(n_neighbors=5),
                'params': {}
            },
            'SVM': {
                'model': SVC(kernel='rbf', probability=True, random_state=random_state),
                'params': {}
            }
        }

        best_model = None
        best_accuracy = 0
        best_model_name = None
        results_summary = {}

        # Train all models and find the best one
        for model_name, config in models_config.items():
            print(f"Training {model_name}...")
            try:
                model = config['model']
                model.fit(X_train_processed, y_train)
                preds = model.predict(X_test_processed)
                accuracy = accuracy_score(y_test, preds)
                results_summary[model_name] = accuracy

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = model_name

                print(f"{model_name} Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Error training {model_name}: {e}")

        self.model = best_model
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test

        # Generate predictions and evaluation metrics
        preds = self.model.predict(X_test_processed)

        results = {
            "best_model": best_model_name,
            "accuracy": accuracy_score(y_test, preds),
            "report": classification_report(y_test, preds, output_dict=True),
            "all_models_results": results_summary,
            "test_size": test_size,
            "training_mode": "automatic"
        }

        self.training_history = results
        print(f"\nBest Model: {best_model_name}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"\nAll Models Performance:\n{pd.DataFrame(list(results_summary.items()), columns=['Model', 'Accuracy'])}")
        
        return results

    def train_model_advanced(self, target_column, model_type="LogisticRegression", test_size=0.2, 
                            random_state=42, hyperparameters=None, scale=True, perform_gridsearch=False, **kwargs):
        """Train a model with advanced custom settings and optional hyperparameter tuning"""
        if self.dataframe is None:
            raise ValueError("No data loaded. Please load data before training a model.")
        
        if target_column not in self.dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        self.target_column = target_column
        print(f"Starting advanced model training...")
        print(f"Model Type: {model_type}")
        print(f"Custom Settings: {kwargs}")

        # Split the dataset
        X = self.dataframe.drop(columns=[target_column])
        y = self.dataframe[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Preprocess data
        X_train_processed = self._preprocess_data(X_train.copy(), y_train, scale=scale)
        X_test_processed = self._preprocess_data(X_test.copy(), y_test, scale=scale)

        # Initialize model with custom hyperparameters
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(max_iter=500, random_state=random_state, **kwargs)
        elif model_type == "DecisionTree":
            self.model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        elif model_type == "RandomForest":
            self.model = RandomForestClassifier(n_jobs=-1, random_state=random_state, **kwargs)
        elif model_type == "KNN":
            self.model = KNeighborsClassifier(**kwargs)
        elif model_type == "SVM":
            self.model = SVC(probability=True, random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Hyperparameter tuning with GridSearchCV if specified
        if perform_gridsearch and hyperparameters:
            print("Performing GridSearchCV for hyperparameter tuning...")
            grid_search = GridSearchCV(self.model, hyperparameters, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train_processed, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters found: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
        else:
            # Train the model without tuning
            print(f"Training {model_type}...")
            self.model.fit(X_train_processed, y_train)

        # Store training data
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test

        # Generate predictions and evaluation metrics
        preds = self.model.predict(X_test_processed)

        results = {
            "model_type": model_type,
            "accuracy": accuracy_score(y_test, preds),
            "report": classification_report(y_test, preds, output_dict=True),
            "test_size": test_size,
            "training_mode": "advanced",
            "hyperparameters_used": kwargs,
            "gridsearch_performed": perform_gridsearch
        }

        self.training_history = results
        print(f"\nModel trained using {model_type}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, preds)}")
        
        return results

    def train_model(self, target_column, model_type="LogisticRegression", test_size=0.2, random_state=42, **kwargs):
        """Legacy method for backward compatibility"""
        return self.train_model_advanced(target_column, model_type, test_size, random_state, **kwargs)
    
    def model_save(self, file_path):
        if self.model is None:
            raise ValueError("No model trained. Please train a model before saving.")
        
        model_data = {
            'model': self.model,
            'target_column': self.target_column,
            'scaler': self.scaler,
            'training_history': self.training_history
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.target_column = model_data['target_column']
        self.scaler = model_data.get('scaler')
        self.training_history = model_data.get('training_history', {})
        print(f"Model loaded from {file_path}")
    
    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        if self.model is None:
            raise ValueError("No model trained. Please train a model before plotting.")
        
        preds = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
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
        fpr, tpr, _ = roc_curve(self.y_test, preds_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
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
    
    def list_available_models(self, directory="models", extension=".pkl"):
        """List all saved model files in a directory."""
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            return []
        
        models = [f for f in os.listdir(directory) if f.endswith(extension)]
        print(f"Found {len(models)} saved models in '{directory}': {models}")
        return models

    def export_training_report(self, file_path="training_report.json"):
        """Export training history and results as JSON"""
        if not self.training_history:
            raise ValueError("No training history available. Train a model first.")
        
        with open(file_path, 'w') as f:
            json.dump(self.training_history, f, indent=4, default=str)
        print(f"Training report exported to {file_path}")


# Example usage:
if __name__ == "__main__":
    manager = AIModelManager()
    
    # Load data
    manager.load_data("your_dataset.csv")
    print(manager.get_dataset_info())
    
    # Option 1: Automatic mode - best model selected automatically
    print("\n=== AUTOMATIC MODE ===")
    auto_results = manager.train_model_automatic(target_column="target", test_size=0.2)
    
    # Option 2: Advanced mode with custom hyperparameters
    print("\n=== ADVANCED MODE ===")
    advanced_results = manager.train_model_advanced(
        target_column="target",
        model_type="RandomForest",
        test_size=0.2,
        n_estimators=200,
        max_depth=15,
        min_samples_split=5
    )
    
    # Option 3: Advanced mode with hyperparameter tuning
    print("\n=== ADVANCED MODE WITH GRIDSEARCH ===")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    gridsearch_results = manager.train_model_advanced(
        target_column="target",
        model_type="RandomForest",
        test_size=0.2,
        hyperparameters=param_grid,
        perform_gridsearch=True
    )
    
    # Save and visualize
    manager.model_save("trained_model.pkl")
    manager.plot_confusion_matrix()
    manager.plot_roc_curve()
    manager.plot_learning_curve()
    manager.export_training_report()