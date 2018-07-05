# -*- coding: utf-8 -*-
import re
from files import emotions
from files import emoji
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

emoji_list = [line.rstrip('\n') for line in open('files/emoji.txt', encoding='UTF-8')]
dictionary = [line.rstrip('\n') for line in open('files/dict.txt', encoding='UTF-8')]


def escape_emoji(text):
    for emoji in emoji_list:
        text = text.replace(emoji, ' ' + emoji + ' ')
    return text


def replace_emoji(text):
    uni_sent = str(text)
    for key in emotions.emoticon_dict.keys():
        uni_sent = uni_sent.replace(key, emotions.emoticon_dict[key])
    utf_sent = re.sub("\xf0...", 'emoticon', uni_sent)
    return utf_sent


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
        # expanded = " ".join([a for a in re.split('([A-Z][a-z]+)', hashtag) if a])
        # text = text.replace(hashtag, expanded)
        # if hashtag == expanded:
        processed_hashtag = hashtag[1:]
        if processed_hashtag in dictionary:
            text = text.replace(hashtag, processed_hashtag)
            # else:
            #     text = text.replace(hashtag, ' ')

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


def escape_text(x, emoji2word=False):
    max_len = 0
    for i in range(len(x)):

        x[i] = escape_emoji(x[i])

        x[i] = emoticon_to_emoji(x[i])

        x[i] = process_hashtags(x[i])

        # x[i] = process_emoji(x[i])

        if emoji2word:
            x[i] = replace_emoji(x[i])

        x[i] = escape_chars(x[i])

        if len(x[i].split()) > max_len:
            max_len = len(x[i].split())

    return x, max_len


def preprocessing_pipeline(train_x, trial_x, test_x, emoji2word=False):
    train_x, max_len_train = escape_text(train_x, emoji2word)
    trial_x, max_len_trial = escape_text(trial_x, emoji2word)
    test_x, max_len_test = escape_text(test_x, emoji2word)

    return train_x, trial_x, test_x, max(max_len_train, max_len_trial, max_len_test)

def preprocess_through_ekphrasis(train_file_path, test_file_path, trial_file_path):
    text_processor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=True,
        spell_correction=True,
        all_caps_tag="wrap",
        fix_bad_unicode=True,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    )

    for file_path in [train_file_path, test_file_path, trial_file_path]:
        with open(file_path, 'r', newline='') as file:
            new_sentences = list()
            labels = list()
            for line in file:
                labels.append(line.split('\t')[0])
                new_sentences.append(" ".join(text_processor.pre_process_doc(line.split('\t')[1])))
        with open(file_path[:-4]+"_ekphrasis.csv", 'w', newline='') as new_file:
            for label, sentence in zip(labels, new_sentences):
                new_file.write("{}\t{}\n".format(label, sentence.replace("[ <hashtag> triggerword </hashtag> #]", "[#TRIGGERWORD#]").replace("[ <allcaps> newline </allcaps> ]", "[NEWLINE]")))
