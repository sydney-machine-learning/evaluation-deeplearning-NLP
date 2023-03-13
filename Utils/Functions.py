import operator

import pandas as pd
import numpy as np
import contractions
from sklearn.model_selection import train_test_split
from tqdm._tqdm_notebook import tqdm_notebook as tqdm


# data cleaning
def data_cleaning(dataset):
    # Remove the html tags
    dataset['Tweet'] = dataset['Tweet'].str.replace('<.*?>', '')
    dataset['Tweet'] = dataset['Tweet'].str.lower()

    # Replace the contractions
    dataset['Tweet'] = dataset['Tweet'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    dataset['Tweet'] = dataset['Tweet'].apply(lambda x: ' '.join(x))
    return dataset

# split the data into train and test
def split_data(dataset, train_size, random_state):
    train, test = train_test_split(dataset, train_size=train_size, random_state=random_state)
    return train, test

# Glove embedding twitter.27B 200d
def glove_embedding():
    vocab, embeddings = [], []
    with open('../../GloVe Embeddings/glove.840B.300d.txt', 'r', encoding='utf-8') as f:
        full_content = f.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')

    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)  # embedding for '<unk>' token.

    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
    return embs_npa, vocab_npa


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def build_vocab(sentences, verbose=True):
    vocab = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1
    return vocab


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, encoding="utf-8") as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix