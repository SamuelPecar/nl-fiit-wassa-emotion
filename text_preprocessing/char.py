import re


def currency_replace(text):
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"£", " pound ", text)
    text = re.sub(r"€", " euro ", text)
    text = re.sub(r"¥", " yen ", text)
    text = re.sub(r"[¢₡₱₭₦]", " currency ", text)

    return text


def char_removing(text):
    text = text.replace("http://url.removed", "")
    text = text.replace("@USERNAME", " @USERNAME ")
    text = re.sub(r"[ं-ో̇]", "", text)

    return text


def char_replacing(text):
    text = re.sub(r"[‘´’̇]+", "\'", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub("\[NEWLINE\]", " ", text)
    text = re.sub("http://url.removed", " ", text)

    return text

def char_escape(text):
    text = re.sub(r"[‼.,?!…*]", " \1 ", text)

    return text
