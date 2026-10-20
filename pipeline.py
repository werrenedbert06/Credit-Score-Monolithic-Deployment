from pathlib import Path
from data_ingestion import DataIngestion
from training import CreditScoreTrainer
from evaluating import CreditScoreEvaluator

class CreditScorePipeline:
    """Main Pipeline: ingesti data, training, evaluasi, dan keputusan deployment."""
    
    def __init__(self, raw_data_path: str | Path, accuracy_threshold: float = 0.70):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.accuracy_threshold = accuracy_threshold
        
        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = CreditScoreTrainer()
        self.evaluator = CreditScoreEvaluator()
    
    def execute(self):
        print("🚀 Memulai Pipeline Klasifikasi Credit Score...")
        
        # 1. Ingestion
        ingested_file = self.ingestor.run()
        
        # 2. Training semua model (dapatkan run_ids)
        run_ids, X_test, y_test = self.trainer.run(ingested_file)
        
        # 3. Evaluasi semua model (load dari MLflow)
        eval_results = self.evaluator.run(run_ids, X_test, y_test)
        
        # 4. Pilih model terbaik berdasarkan accuracy
        best_model = max(eval_results.items(), key=lambda x: x[1]["accuracy"])
        best_name, best_metrics = best_model
        print(f"\n🏆 Model terbaik: {best_name} (Accuracy = {best_metrics['accuracy']:.4f})")
        
        # 5. Keputusan deployment
        print("\n--- Keputusan Deployment ---")
        if best_metrics["accuracy"] >= self.accuracy_threshold:
            print(f"🎉 Disetujui: Akurasi {best_metrics['accuracy']:.3f} >= threshold {self.accuracy_threshold}")
            print(f"   Model {best_name} siap diproduksi.")
        else:
            print(f"❌ Ditolak: Akurasi {best_metrics['accuracy']:.3f} < threshold {self.accuracy_threshold}")
            print("   Ulangi eksperimen dengan tuning lebih lanjut.")
    
if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_C.csv"
    pipeline = CreditScorePipeline(raw_data_path=DATA_INPUT, accuracy_threshold=0.70)
    pipeline.execute()