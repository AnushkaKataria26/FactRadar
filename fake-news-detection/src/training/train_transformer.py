import time
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.transformer_tokenize import get_tokenizer
from src.training.prepare_transformer_dataset import prepare_transformer_datasets

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1': f1
    }

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset, val_dataset = prepare_transformer_datasets()
    tokenizer = get_tokenizer()
    
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    batch_size = 16
    training_args = TrainingArguments(
        output_dir="models/v0.1_transformer_checkpoints",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42,
        report_to="none" # Disable wandb/mlflow for now unless configured
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    start_time = time.time()
    try:
        print(f"Starting training with batch size {batch_size}...")
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("GPU OutOfMemoryError caught. Halving batch size and retrying once...")
        torch.cuda.empty_cache()
        batch_size = 8
        training_args.per_device_train_batch_size = batch_size
        training_args.per_device_eval_batch_size = batch_size
        
        # Re-initialize trainer with new args
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
        print(f"Restarting training with batch size {batch_size}...")
        train_result = trainer.train()
        
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Save Model
    save_path = "models/v0.1_transformer"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Evaluate final model
    eval_metrics = trainer.evaluate()
    
    # Save Metrics
    metrics_data = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime()),
        "model_name": model_name,
        "batch_size_used": batch_size,
        "training_time_seconds": training_time,
        "device": str(device),
        "final_val_accuracy": eval_metrics.get("eval_accuracy"),
        "final_val_f1_macro": eval_metrics.get("eval_f1"),
        "final_val_precision_macro": eval_metrics.get("eval_precision_macro"),
        "final_val_recall_macro": eval_metrics.get("eval_recall_macro"),
    }
    
    metrics_path = "models/v0.1_transformer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train()
