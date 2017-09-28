import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def findTopic(ls):
    topic_score_ls = list()
    for i in ls:
        topic_score_ls.append(i[1])
        max_value = max(topic_score_ls)
        max_index = topic_score_ls.index(max_value)
    return max_index

def lsi_find_topicMatrix(file, n_topic):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    texts = []

    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        texts.append(line)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topic)
    corpus_lsi = lsi[corpus_tfidf]

    topic_matrix = np.empty([len(texts), n_topic])
    i = 0

    for doc in corpus_lsi:
        topic_matrix[i] = [item[1] for item in doc]
        i +=1

    return topic_matrix


def lsi_find_topicLabel(file, n_topic):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    texts = []

    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        texts.append(line)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topic)
    corpus_lsi = lsi[corpus_tfidf]

    lable = list()


    for doc in corpus_lsi:
        lable.append(findTopic(doc))

    return lable

def lda_findTopic_probMatrix (file, n_topic):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    texts = []

    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        texts.append(line)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

    mm = corpora.MmCorpus('/tmp/deerwester.mm')
    id2word = corpora.Dictionary.load('/tmp/deerwester.dict')

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=n_topic, update_every=1, chunksize=10000, passes=1)

    corpus_lda = lda[corpus_tfidf]

    topic_matrixProb = np.empty([len(texts), n_topic])
    i = 0

    for doc in corpus_lda:
        topic_matrixProb[i] = [item[1] for item in doc]
        i += 1

    return topic_matrixProb


def lda_findTopic_label(file, n_topic):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    texts = []

    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        texts.append(line)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

    mm = corpora.MmCorpus('/tmp/deerwester.mm')
    id2word = corpora.Dictionary.load('/tmp/deerwester.dict')

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=n_topic, update_every=1, chunksize=10000,
                                   passes=1)

    corpus_lda = lda[corpus_tfidf]

    lable = list()

    for doc in corpus_lda:
        lable.append(findTopic(doc))

    return lable


def lis_topic_words(file, n_topic, n_words):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    texts = []

    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        texts.append(line)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)  # initialize an LSI transformation

    return lsi.print_topics(n_topic, n_words)


def lda_topic_words(file, n_topic,n_words):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    texts = []

    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        texts.append(line)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

    mm = corpora.MmCorpus('/tmp/deerwester.mm')
    id2word = corpora.Dictionary.load('/tmp/deerwester.dict')
    print(mm)

    lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=n_topic, update_every=1, chunksize=10000, passes=1)
    return lda.print_topics(n_topic,n_words)



