import os


def read_and_parse_word_clusters():
    DEFAULT_FILENAME = os.getcwd() + os.sep + 'twitter_word_clusters' + os.sep + 'input' + os.sep + '50mpaths2.txt'

    with open(DEFAULT_FILENAME, 'r') as f:
        lines = f.readlines()

    word_clusters = dict()
    for l in lines:
        tokens = l.split('\t')
        cluster_path = tokens[0]
        word = tokens[1]

        word_clusters[word] = cluster_path

    return word_clusters


def tokenise(clusters, sentence):
    vector = dict()

    for w in sentence.split(' '):
        if w in clusters:
            path = clusters[w]
            if path in vector:
                vector[path] += 1
            else:
                vector[path] = 1

    return vector

clusters = read_and_parse_word_clusters()
sample_text = 'A not so happy Halloween for Chipotle. 43 restaurants in Oregon and WA. temporarily shut down as an e-coli outbreak is linked to the chain.'
dict_vector = tokenise(clusters, sample_text)

print(dict_vector)
