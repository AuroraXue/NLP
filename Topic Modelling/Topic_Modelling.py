import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from gensim import corpora, models
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

f = open('topics.txt', 'r')
lines = f.readlines()
f.close()
start = False
start_index =list()
end_index = list()
c = 0

for i, line in enumerate(lines):
    if line.startswith('<narr>'):
        c=c+1
        start = True
        start_index.append(i)
    if start and line.startswith('</top>'):
        end_index.append(i)
        start = False

select_line = []

for x,y in zip(start_index, end_index):
    line = lines[(x+1):y]
    line = ' '.join(line)
    line = re.sub('[^a-zA-Z.]', ' ', line)
    line = line.lower().split()
    line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
    line = ' '.join(line)
    select_line.append(line)

thefile = open('cleaned_sentence.txt', 'w')
for line in select_line:
    thefile.write("%s\n" % line)
thefile.close()

# Example code:
n_sample = 2000
n_features = 1000
n_topic = 10

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers','footers','quotes'))
data_sample = dataset.data[:n_sample] # the same as the selected line

tf_vectorizer = CountVectorizer(max_df=0.95, min_df= 2, max_features= n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data_sample)

lda = LatentDirichletAllocation(n_topics = n_topic, max_iter= 5, learning_method='online', learning_offset=50., random_state=0)

lda.fit(tf)


# Come to my part:
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = CountVectorizer(max_features=600, analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
matrix =  vectorizer.fit_transform(select_line)
feature_names = vectorizer.get_feature_names()

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(select_line)

lda = LatentDirichletAllocation(n_topics=8, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
model = lda.fit(tf)

tf_feature_names = tf_vectorizer.get_feature_names()



import lda
import numpy as np

vocab = feature_names
model = lda.LDA(n_topics=8, n_iter=500, random_state=1)
model.fit(matrix)
topic_word = model.topic_word_
n_top_words = 20

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))

for n in range(50):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}".format(n, topic_most_pr))






# genism code:
texts = [[word for word in document.lower().split() ]
         for document in select_line]
# Create Dictionary.
id2word = corpora.Dictionary(texts)
# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=8, update_every=1, chunksize=10000, passes=1)

for top in lda.print_topics():
  print(top)



########################## New Code ##############################
# Importing Gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora

# tf_vectorizer = TfidfVectorizer(input='select_line', analyzer='word',
#                      min_df=0, stop_words='english', sublinear_tf=True)
#
# tf = tf_vectorizer.fit_transform(select_line)


doc_clean = [doc.split() for doc in select_line]
# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=8, id2word = dictionary, passes=50)

for top in lda.print_topics(num_words=20):
  print(top)


vec_bow = dictionary.doc2bow(select_line[0].lower().split())
lsi = models.LsiModel(mm, id2word=dictionary, num_topics=8)
vec_lsi = lsi[vec_bow]
print(vec_lsi)

ldamodel.doc_topic_

