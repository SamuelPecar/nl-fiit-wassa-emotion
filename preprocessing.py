# -*- coding: utf-8 -*-
import re

emoji_list = [line.rstrip('\n') for line in open('files/emoji.txt')]


def escape_emoji(text):
    for emoji in emoji_list:
        text = text.replace(emoji, ' ' + emoji + ' ')
    return text


def escape_chars(text):
    text = re.sub(r"\s", " ", text)
    text = text.replace("‼", " ‼ ")
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("…", " … ")
    text = text.replace("*", " * ")
    text = text.replace("[NEWLINE]", ". ")
    text = text.replace("http://url.removed", "")
    text = text.replace("@USERNAME", "")

    text = re.sub(r"''", " \" ", text)
    text = re.sub(r"[”“❝„\"‘´’]", " \' ", text)

    text = re.sub(" '", " \' ", text)
    text = re.sub("' ", " \' ", text)
    text = re.sub(r"^'", " \' ", text)
    text = re.sub(r"'$", " \' ", text)

    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"can't", " cannot ", text)
    text = re.sub(r"n't", " not ", text)

    return text


def escape_text(x):
    for i in range(len(x)):
        x[i] = escape_emoji(x[i])
        x[i] = escape_chars(x[i])

    return x


def preprocessing_pipeline(train_x, test_x):
    train_x = escape_text(train_x)
    test_x = escape_text(test_x)

    max_len_train = len(max(train_x[:], key=len).split())
    max_len_test = len(max(test_x[:], key=len).split())

    return train_x, test_x, max_len_train if max_len_train > max_len_test else max_len_test
