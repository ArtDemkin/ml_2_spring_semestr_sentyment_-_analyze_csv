import re
import altair as alt
import streamlit as st
import pandas as pd
from nltk import ngrams
import pymorphy2
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="Main_diag", page_icon="🏠")  # Полнооконное представление приложения
alt.themes.enable('streamlit')
st.write("# Приложение для определения наиболее часто встречающихся пар слов в строке csv файла")
st.sidebar.success("Меню приложения")
number = st.number_input('Укажите количество пар слов, которые нужно отобразить:',
                         min_value=10, max_value=100)
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with open("out.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.write("File saved!")


def count_word_pairs(text):
    """
    Вспомогательная функция, принимает на вход Series и возвращает количество пар значений Series
    """
    pairs = ngrams(text.split(), 2)
    return pd.Series(pairs).value_counts()


morph = pymorphy2.MorphAnalyzer()


def lemmatize_word(word):
    """
Вспомогательная мультиязыковая(Rus,En) функция для лемматизауии слов файла из 544 строк
    """
    return morph.parse(word)[0].normal_form


def words_pair():
    """
Основная функция для определения наиболее часто встречающихся пар слов
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf8', sep='/n', engine='python', header=None)
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
        n_largest_cols = df_freq.count().nlargest(number + 1)
        n_largest_cols.to_csv('имя_файла.csv')
        df = pd.read_csv('имя_файла.csv', encoding='utf8', sep=',', engine='python')
        df = df.drop([0])
        df[['Unnamed: 1', 'Unnamed: 2']] = df['Unnamed: 0'].str.split(' ', expand=True)
        df = df.drop('Unnamed: 0', axis=1)
        df['Unnamed: 1'] = df['Unnamed: 1'].str.replace('[^\w\s]+', '')
        df['Unnamed: 2'] = df['Unnamed: 2'].str.replace('[^\w\s]+', '')
        # Create heatmap using plotly.express
        df_pivoted = df.pivot(index='Unnamed: 1', columns='Unnamed: 2', values='0')
        # Create heatmap using plotly.express
        fig = px.imshow(df_pivoted, x=df_pivoted.columns, y=df_pivoted.index)
        fig.update_layout(title='Пары слов по частоте встречаемости в строке')
        st.plotly_chart(fig)


print(words_pair())
