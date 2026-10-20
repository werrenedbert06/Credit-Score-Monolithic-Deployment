import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CreditScoreEvaluator:
    """Evaluasi semua model berdasarkan run_id, load model dari MLflow, dan log metrik."""
    
    def run(self, run_ids, X_test, y_test):
        print("\n=== Evaluasi Model ===")
        results = {}
        
        for model_name, run_id in run_ids.items():
            print(f"Evaluating {model_name} (run_id: {run_id})...")
            
            # Load model dari MLflow menggunakan run_id
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            
            y_preds = model.predict(X_test)
            acc = accuracy_score(y_test, y_preds)
            prec = precision_score(y_test, y_preds, average="macro")
            rec = recall_score(y_test, y_preds, average="macro")
            f1 = f1_score(y_test, y_preds, average="macro")
            
            # Log metrik ke run yang sama (aktifkan konteks run)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
            
            results[model_name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
            print(f"{model_name} → Accuracy = {acc:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f} | F1 Score = {f1:.4f}")
        
        return results