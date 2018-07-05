# -*- coding: utf-8 -*-
import re
from files import emoji

emoji_list = [line.rstrip('\n') for line in open('files/emoji.txt', encoding='UTF-8')]
dictionary = [line.rstrip('\n') for line in open('files/dict.txt', encoding='UTF-8')]


def escape_emoji(text):
    for emoji in emoji_list:
        text = text.replace(emoji, ' ' + emoji + ' ')
    return text


def emoticon_to_emoji(text):
    text = re.sub(r":\)", " 🙂 ", text)
    text = re.sub(r":-\)", " 🙂 ", text)
    text = re.sub(r":D", " 😀 ", text)
    text = re.sub(r":-D", " 😀 ", text)
    text = re.sub(r":\(", " 🙁 ", text)
    text = re.sub(r":-\(", " 🙁 ", text)
    text = re.sub(r";\)", " 😉 ", text)
    text = re.sub(r";-\)", " 😉 ", text)

    return text


def process_emoji(text):
    for e in emoji.emoji_dict:
        text = text.replace(e, emoji.emoji_dict[e])
    # return re.sub("\xf0...", '', str(text))
    return text


def process_hashtags(text):
    hashtags = re.findall(r" (#\w+)", text)

    for hashtag in hashtags:
        processed_hashtag = hashtag[1:]
        if processed_hashtag in dictionary:
            text = text.replace(hashtag, processed_hashtag)

    return text


def escape_chars(text):
    text = text.replace("[NEWLINE]", ". ")
    text = text.replace("http://url.removed", "")
    text = text.replace("@USERNAME", "")

    text = re.sub(r"\s", " ", text)
    text = re.sub(r"[‘´’]", "\'", text)
    text = re.sub(r"[”“❝„\"]", " ", text)
    text = re.sub("/", " / ", text)

    text = re.sub(r"[‼.,?!…*]", " ", text)
    text = re.sub(r"[\(\)~+=<>{}:;\-—|_\^]", " ", text)
    text = re.sub(r"[0-9]", " ", text)

    # text = text.replace("‼", " ‼ ")
    # text = text.replace(".", " . ")
    # text = text.replace(",", " , ")
    # text = text.replace("!", " ! ")
    # text = text.replace("?", " ? ")
    # text = text.replace("…", " … ")
    # text = text.replace("*", " * ")

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
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"£", " pound ", text)
    text = re.sub(r"€", " euro ", text)

    return text


def preprocessing(x):
    max_len = 0
    for i in range(len(x)):

        x[i] = escape_emoji(x[i])
        x[i] = emoticon_to_emoji(x[i])
        x[i] = process_hashtags(x[i])
        x[i] = process_emoji(x[i])
        x[i] = escape_chars(x[i])

        if len(x[i].split()) > max_len:
            max_len = len(x[i].split())

    return x, max_len


def preprocessing_pipeline(train_x, trial_x, test_x):
    train_x, max_len_train = preprocessing(train_x)
    trial_x, max_len_trial = preprocessing(trial_x)
    test_x, max_len_test = preprocessing(test_x)

    return train_x, trial_x, test_x, max(max_len_train, max_len_trial, max_len_test)
