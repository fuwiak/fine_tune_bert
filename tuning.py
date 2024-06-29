import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def load_data(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def fine_tune_model(data_file, model_name, output_dir, num_train_epochs=3, per_device_train_batch_size=8,
                    per_device_eval_batch_size=8):
    # Load dataset
    dataset = load_data(data_file)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy="epoch",
        save_total_limit=1,
        logging_dir='./logs',
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Fine-tune BERT model for sentiment analysis")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the CSV file containing training data")
    parser.add_argument("--model_name", type=str, default="dkleczek/bert-base-polish-uncased-v1",
                        help="Name of the pretrained BERT model")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation")

    args = parser.parse_args()

    fine_tune_model(
        data_file=args.data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size
    )
