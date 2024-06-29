import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def load_fine_tuned_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer

def main():
    # Directory where the fine-tuned model is saved
    model_dir = "."

    # Load the fine-tuned model and tokenizer
    sentiment_analyzer = load_fine_tuned_model(model_dir)

    # Example text for sentiment analysis
    example_text = "To jest świetny przykład pozytywnego tekstu."

    # Analyze sentiment
    results = sentiment_analyzer(example_text)

    # Print results
    for result in results:
        print(f"Label: {result['label']}, Score: {result['score']:.2f}")

if __name__ == "__main__":
    main()
