#!/usr/bin/env python3
"""
Upload fine-tuning data to Cohere and start training job

Usage:
    python scripts/upload_cohere_finetune.py

Requires:
    - COHERE_API_KEY in .env
    - cohere Python package: pip install cohere
"""

import os
import time
import cohere
from pathlib import Path
from dotenv import load_dotenv
from cohere.models import FinetunedModel, BaseModel, Settings

load_dotenv()

# Configuration
TRAIN_FILE = "experiments/05-finetune/cohere_rerank_data/train.jsonl"
VAL_FILE = "experiments/05-finetune/cohere_rerank_data/validation.jsonl"
FINETUNE_NAME = "multiturn-rag-rerank"

def main():
    print("=" * 80)
    print("  COHERE FINE-TUNING UPLOAD")
    print("=" * 80)
    print()
    
    # Initialize Cohere client
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY not found in .env")
        return
    
    co = cohere.Client(api_key=api_key)
    print("SUCCESS: Cohere client initialized")
    print()
    
    # Verify files exist
    train_path = Path(TRAIN_FILE)
    val_path = Path(VAL_FILE)
    
    if not train_path.exists():
        print(f"ERROR: Training file not found: {TRAIN_FILE}")
        print("   Run: python scripts/generate_cohere_finetune_data.py first")
        return
    
    if not val_path.exists():
        print(f"ERROR: Validation file not found: {VAL_FILE}")
        return
    
    print(f"Training file: {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")
    print(f"Validation file: {val_path} ({val_path.stat().st_size / 1024:.1f} KB)")
    print()
    
    # Count examples
    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    with open(val_path) as f:
        val_count = sum(1 for _ in f)
    
    print(f"Dataset Statistics:")
    print(f"   Training examples: {train_count}")
    print(f"   Validation examples: {val_count}")
    print()
    
    # Step 1: Upload training dataset
    print("Step 1: Uploading training dataset...")
    try:
        train_dataset = co.datasets.create(
            name=f"{FINETUNE_NAME}-train",
            data=open(train_path, 'rb'),
            type="reranker-finetune-input"
        )
        print(f"   Training dataset uploaded: {train_dataset.id}")
        
        # Wait for dataset to be ready
        print(f"   Waiting for dataset to be ready...")
        train_dataset = co.wait(train_dataset)
        print(f"   Dataset status: {train_dataset.validation_status}")
    except Exception as e:
        print(f"   ERROR uploading training data: {e}")
        return
    
    # Step 2: Upload validation dataset (optional but recommended)
    print()
    print("Step 2: Uploading validation dataset...")
    try:
        val_dataset = co.datasets.create(
            name=f"{FINETUNE_NAME}-validation",
            data=open(val_path, 'rb'),
            type="reranker-finetune-input"
        )
        print(f"   Validation dataset uploaded: {val_dataset.id}")
        
        # Wait for dataset to be ready
        print(f"   Waiting for dataset to be ready...")
        val_dataset = co.wait(val_dataset)
        print(f"   Dataset status: {val_dataset.validation_status}")
        
        has_validation = True
    except Exception as e:
        print(f"   WARNING: Could not upload validation data: {e}")
        print(f"   Continuing without validation dataset...")
        has_validation = False
    
    print()
    
    # Step 3: Create fine-tuning job
    print("Step 3: Creating fine-tuning job...")
    try:
        settings_dict = {
            "base_model": BaseModel(
                base_type="BASE_TYPE_RERANK"
            ),
            "dataset_id": train_dataset.id
        }
        
        # Add validation dataset if available
        if has_validation:
            settings_dict["eval_data"] = {"dataset_id": val_dataset.id}
        
        create_response = co.finetuning.create_finetuned_model(
            request=FinetunedModel(
                name=FINETUNE_NAME,
                settings=Settings(**settings_dict)
            )
        )
        
        model_id = create_response.finetuned_model.id
        status = create_response.finetuned_model.status
        
        print(f"   Fine-tuning job created successfully!")
        print()
        print("=" * 80)
        print("  FINE-TUNING JOB DETAILS")
        print("=" * 80)
        print()
        print(f"Model ID: {model_id}")
        print(f"Model Name: {FINETUNE_NAME}")
        print(f"Status: {status}")
        print()
        print("Training will take approximately 2-4 hours.")
        print()
        print("Monitor progress:")
        print(f"  1. Cohere Dashboard: https://dashboard.cohere.com/fine-tuning")
        print(f"  2. Check status via Python:")
        print(f"     co.finetuning.get_finetuned_model('{model_id}')")
        print()
        
        # Step 4: Monitor training progress
        print("Step 4: Monitoring training progress...")
        print("(Press Ctrl+C to stop monitoring - training will continue)")
        print()
        
        try:
            while True:
                model_status = co.finetuning.get_finetuned_model(model_id)
                current_status = model_status.status
                print(f"   Current status: {current_status}")
                
                if current_status == "READY":
                    print()
                    print("=" * 80)
                    print("  TRAINING COMPLETE!")
                    print("=" * 80)
                    print()
                    print(f"Fine-tuned model is ready to use!")
                    print()
                    print("Your fine-tuned model ID:")
                    print(f"  {model_id}")
                    print()
                    print("Update experiment configs:")
                    print(f"  sed -i 's/REPLACE_WITH_FINETUNED_MODEL_ID/{model_id}/g' \\")
                    print(f"    configs/experiments/05-finetune/*.yaml")
                    print()
                    print("Test your model:")
                    print(f"  response = co.rerank(")
                    print(f"      query='your query',")
                    print(f"      documents=['doc1', 'doc2'],")
                    print(f"      model='{model_id}'")
                    print(f"  )")
                    print()
                    print("=" * 80)
                    break
                
                elif current_status == "FAILED":
                    print()
                    print("ERROR: Training failed!")
                    print("Check Cohere Dashboard for details:")
                    print("https://dashboard.cohere.com/fine-tuning")
                    break
                
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print()
            print("Monitoring stopped. Training continues in background.")
            print(f"Check status at: https://dashboard.cohere.com/fine-tuning")
            print()
        
    except Exception as e:
        print(f"   ERROR creating fine-tuning job: {e}")
        print()
        print("Note: You may need to use the Cohere Dashboard instead:")
        print("  https://dashboard.cohere.com/fine-tuning")
        print()
        print(f"Files ready for manual upload:")
        print(f"  - {TRAIN_FILE}")
        print(f"  - {VAL_FILE}")
        return

if __name__ == "__main__":
    main()
