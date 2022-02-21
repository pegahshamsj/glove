from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import string
import nltk
import re


nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    '''stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]'''
    
    text = " ".join(text)

    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    ## Stemming
    text = text.split()
    lemma = nltk.wordnet.WordNetLemmatizer()
    lematized_words = [lemma.lemmatize(word) for word in text]
    text = " ".join(lematized_words)
    print(text)
    
    return text


def glove_embedding(title):
    title = [clean_text(title)]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(title)

    EMBEDDING_FILE = 'glove.6B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    embedding_matrix = np.zeros((400000, 300))
    for word, index in tokenizer.word_index.items():
        if index > 400000 - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    
    print(embedding_matrix[i])


def main():
    title = 'title of the video'
    glove_embedding(title)


if __name__ == "__main__":
    main()
