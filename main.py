import csv
import re


def text_preprocess():
    with open('file.csv', 'r', newline='', encoding='utf8') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    for row in data:
        for i in range(len(row)):
            row[i] = row[i].lower()
            row[i] = re.sub(r'[^\w\s]', '', row[i]) # пунктуация
            row[i] = row[i].replace(',', ';')
            row[i] = re.sub(r'((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,'
                            r'5}(?:com|co|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly'
                            r'|sm){1}(?:\/['
                            r'a-zA-Z0-9]{1,})*)', '', row[i])  # ссылки




    with open('имя_файла.csv', 'w', newline='', encoding='utf8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

print(text_preprocess())
