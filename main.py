import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt


def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
    return sentiment_analyzer


def get_label_name(label):
    label_mapping = {
        "LABEL_0": "Negatywny",
        "LABEL_1": "Pozytywny"
    }
    return label_mapping.get(label, label)


def plot_results(results):
    labels = [get_label_name(res['label']) for res in results[0]]
    scores = [res['score'] for res in results[0]]

    fig, ax = plt.subplots()
    ax.bar(labels, scores, color=['red', 'green'])
    ax.set_xlabel('Etykieta')
    ax.set_ylabel('Wynik')
    ax.set_title('Wyniki Analizy Sentymentu')
    st.pyplot(fig)


def main():
    # Ustawienia Streamlit
    st.title("Analiza Sentymentu w Języku Polskim")
    st.write("Wprowadź tekst w języku polskim, aby zbadać jego sentyment za pomocą modelu BERT.")

    # Boczne menu (sidebar)
    st.sidebar.title("Ustawienia")
    model_name = st.sidebar.selectbox(
        "Wybierz model:",
        ["dkleczek/bert-base-polish-uncased-v1", "dkleczek/bert-base-polish-cased-v1"]
    )

    upload_file = st.sidebar.file_uploader("Załaduj plik tekstowy", type=["txt"])

    # Wejście użytkownika
    user_input = st.text_area("Wprowadź tekst tutaj:")

    # Analiza sentymentu
    if st.button("Analizuj"):
        if upload_file:
            text = upload_file.read().decode("utf-8")
            st.write("Załadowany tekst:")
            st.write(text)
        else:
            text = user_input

        if text:
            sentiment_analyzer = load_model(model_name)
            results = sentiment_analyzer(text)
            st.write("Wyniki analizy sentymentu:")
            for result in results[0]:
                label_name = get_label_name(result['label'])
                st.write(f"Label: {label_name}, Score: {result['score']:.2f}")

            plot_results(results)
        else:
            st.write("Proszę wprowadzić tekst do analizy lub załadować plik.")


if __name__ == "__main__":
    main()
