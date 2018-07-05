# -*- coding: utf-8 -*-
import re
from files import emoji
from text_preprocessing import emoji, hashtag, char


def escape_chars(text):
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


def preprocess_text(text):
    text = char.char_removing(text)
    text = char.char_replacing(text)
    text = char.currency_replace(text)
    text = emoji.emoticon_to_emoji(text)
    text = emoji.emoji_gender(text)
    text = re.sub(r"\s+", " ", text)
    text = emoji.emoji_categorization(text)
    text = emoji.escape_emoji(text)
    # text = hashtag.process_hashtags(text)
    text = re.sub(r"\s+", " ", text)

    return text


def preprocessing(x):
    max_len = 0
    for i in range(len(x)):
        x[i] = preprocess_text(x[i])

        if len(x[i].split()) > max_len:
            max_len = len(x[i].split())

    return x, max_len


def preprocessing_pipeline(train_x, trial_x, test_x):
    train_x, max_len_train = preprocessing(train_x)
    trial_x, max_len_trial = preprocessing(trial_x)
    test_x, max_len_test = preprocessing(test_x)

    return train_x, trial_x, test_x, max(max_len_train, max_len_trial, max_len_test)
