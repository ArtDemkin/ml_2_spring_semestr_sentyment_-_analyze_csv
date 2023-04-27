import re
import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import ngrams
import pymorphy2
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="Main_diag", page_icon="🏠")  # Полнооконное представление приложения
st.write("# Приложение для анализа тональности сообщений в чатах Telegram")
number = st.number_input('Укажите количество пар слов, которые нужно отобразить:',
                         min_value=10, max_value=100)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Open and write the contents of the uploaded file to a new file on disk
    with open("out.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.write("File saved!")


def count_word_pairs(text):
    """
    Вспомогательная функция, принимает на вход Series и
    """
    pairs = ngrams(text.split(), 2)
    return pd.Series(pairs).value_counts()


morph = pymorphy2.MorphAnalyzer()


def lemmatize_word(word):
    return morph.parse(word)[0].normal_form


def words_pair():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf8', sep='/n', engine='python')
        d_f = df.squeeze()
        d_f = d_f.str.lower()
        d_f = d_f.str.replace(re.compile(r'http\S+'), '', regex='True')
        d_f = d_f.str.replace(re.compile(r'[^\w\s]+'), '', regex='True')
        d_f = d_f.str.replace(re.compile(r'\d+'), '', regex='True')
        d_f = d_f.str.replace(re.compile(r'\n'), '', regex='True')
        d_f = d_f.str.replace(re.compile(r'\s{2,}'), '', regex='True')
        d_f = d_f.str.replace(re.compile(r'\s{2,}'), '', regex='True')
        d_f = d_f.apply(lambda x: ' '.join([lemmatize_word(w) for w in x.split()]))
        df_freq = d_f.apply(count_word_pairs).reset_index()
        n_largest_cols = df_freq.count().nlargest(number+1)
        n_largest_cols.to_csv('название_файла.csv')
        df = pd.read_csv('название_файла.csv', encoding='utf8', sep=',', engine='python')
        df = df.drop([0])
        df[['Unnamed: 1', 'Unnamed: 2']] = df['Unnamed: 0'].str.split(' ', expand=True)
        df = df.drop('Unnamed: 0', axis=1)
        df['Unnamed: 1'] = df['Unnamed: 1'].str.replace('[^\w\s]+', '')
        df['Unnamed: 2'] = df['Unnamed: 2'].str.replace('[^\w\s]+', '')
        heatmap_data = df.pivot(index='Unnamed: 1', columns='Unnamed: 2', values='0')
        plt.figure(figsize=(number/2, number/3))
        fig = px.density_heatmap(heatmap_data, marginal_x="histogram", marginal_y="histogram")
        fig.show()

print(words_pair())
