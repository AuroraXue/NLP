import re
import nltk
import lda
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer


def get_top_models(file, n_topic, n_words):
    f = open(file, 'r')
    lines = f.readline()
    text_list = []
    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        line = ' '.join(line)
        text_list.append(line)

    vectorizer = CountVectorizer(max_features=600, analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    corpus = vectorizer.fit_transform(text_list)
    feature_names = vectorizer.get_feature_names()

    model = lda.LDA(n_topics=8, n_iter=500, random_state=1)
    model.fit(corpus)

    topic_word = model.topic_word_
    doc_topic = model.doc_topic_

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(feature_names)[np.argsort(topic_dist)][:-n_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    for n in range(50):
        topic_most_pr = doc_topic[n].argmax()
        print("doc: {} topic: {}".format(n, topic_most_pr))



