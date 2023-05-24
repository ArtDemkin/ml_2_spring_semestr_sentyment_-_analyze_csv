import pandas as pd
from transformers import pipeline
import streamlit as st

st.set_page_config(layout="wide", page_title="2__analyze", page_icon="📊")  # Полнооконное представление приложения
st.write("# Просмотр наиболее положительных и отрицательных строк по тональности в csv файле")

number = st.number_input('Укажите количество строк:',
                         min_value=1, max_value=100)
options = ['POSITIVE', 'NEGATIVE']
selected_option = st.selectbox('Choose an option', options)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with open("out.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.write("File saved!")
df = pd.read_csv(uploaded_file, encoding='utf8', sep='/n', engine='python', header=None)

sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")


def sent_str():
    """
Основная функция для определения наиболее положительных и отрицательных строк
    """
    selected_option_rows = []
    for index, row in df.iterrows():
        text = row[0]  # замените 'текст' на имя столбца с текстом в вашем CSV-файле
        sentiment = sentiment_analyzer(text)[0]
        if sentiment['label'] == selected_option:
            selected_option_rows.append((text, sentiment['score']))
    selected_option_rows = sorted(selected_option_rows, key=lambda x: x[1], reverse=True)
    top_positive_rows = selected_option_rows[:number]

    st.write('Топ', number, selected_option, 'строк')
    for i, row in enumerate(top_positive_rows, 1):
        st.write(i, "-", row[0])
        st.write("Оценка тональности:", row[1])


print(sent_str())
