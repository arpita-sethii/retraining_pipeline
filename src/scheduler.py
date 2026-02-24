import schedule
import time
import mlflow
from src.pipeline import run_pipeline

def job():
    print("\n" + "🕐 " * 20)
    print("SCHEDULER: Running pipeline check...")
    print("🕐 " * 20)
    run_pipeline()
    print("\nNext run in 30 seconds...\n")

def run_scheduler():
    mlflow.set_experiment("retraining_pipeline")

    print("=" * 50)
    print("🚀 Automated Retraining Scheduler Started")
    print("   Checking model health every 30 seconds")
    print("   MLflow UI: http://127.0.0.1:5000")
    print("=" * 50)

    # Run immediately on start
    job()

    # Then schedule every 30 seconds
    schedule.every(30).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run_scheduler()