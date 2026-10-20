from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Impor class preprocessor
from preprocessing import CreditScorePreprocessor

class CreditScoreTrainer:
    """Menangani perakitan pipeline, pelatihan multi-model, dan pelacakan MLflow."""
    
    def __init__(self):
        self.experiment_name = "Credit Score Classification"
        self.artifact_dir = Path("artifacts")
        self.preprocessor = CreditScorePreprocessor()
        self.random_state = 42
        

        
        # Inisialisasi Model Registry (Daftar model yang akan ditraining)
        self.model_registry = {
            "RF-5": RandomForestClassifier(n_estimators=200, criterion="entropy", class_weight="balanced", random_state=42, n_jobs=-1),
            "HGB-2": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42),
            "XGB-4": XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, objective='multi:softmax', num_class=3, random_state=42)
        }
        
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)
        
    def run(self, data_path: str | Path):
        print("⚙️ Memulai pra-pemrosesan data...")
        x_train, x_test, y_train, y_test = self.preprocessor.clean_and_split(data_path)
        transformer = self.preprocessor.get_transformer()
        
        run_ids = {}

        # Loop melalui registry model
        for model_name, model_instance in self.model_registry.items():
            print(f"🚀 Training model: {model_name}...")
            
            # Membangun pipeline untuk setiap model
            pipeline = Pipeline([
                ("preprocessing", transformer),
                ("classifier", model_instance)
            ])

            # Start MLflow run per model
            with mlflow.start_run(run_name=model_name) as run:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_type", type(model_instance).__name__)

                # Training
                pipeline.fit(x_train, y_train)
                
                # Simpan ke MLFlow
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
                run_ids[model_name] = run.info.run_id
                
                # Simpan ke file lokal untuk Streamlit
                model_file = self.artifact_dir / f"{model_name}_pipeline.pkl"
                joblib.dump(pipeline, model_file, compress=('lzma', 9))
                print(f"✅ Model {model_name} saved to {model_file}")

        return run_ids, x_test, y_test