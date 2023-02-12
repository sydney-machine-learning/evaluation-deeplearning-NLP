import pandas as pd
import numpy as np
import contractions
from sklearn.model_selection import train_test_split

# data cleaning
def data_cleaning(dataset):
    # Remove the html tags
    dataset['review'] = dataset['review'].str.replace('<.*?>', '')

    # Replace the contractions
    dataset['review'] = dataset['review'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    dataset['review'] = dataset['review'].apply(lambda x: ' '.join(x))

    # sentiment labels
    dataset['sentiment'] = dataset['sentiment'].replace('positive', 1)
    dataset['sentiment'] = dataset['sentiment'].replace('negative', 0)
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


def review_length_distribution(dataset):
    # review length distribution
    dataset['review_length'] = dataset['review'].apply(lambda x: len(x.split()))
    dataset['review_length'].hist()
    return dataset

