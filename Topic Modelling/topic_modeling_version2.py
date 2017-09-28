import re
import nltk
import lda
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from gensim import corpora, models
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
######################## Text Processing ############################
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

texts = [[word for word in document.lower().split() ]
         for document in select_line]

thefile = open('cleaned_sentence.txt', 'w')
for line in select_line:
    thefile.write("%s\n" % line)
thefile.close()


############################### Use Sklearn Package LatentDirichletAllocation ####################################
# By using TF-IDF: https://www.quora.com/How-do-you-combine-LDA-and-tf-idf
'''
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=600,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(select_line)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# Run NMF
nmf = NMF(n_components=8, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
display_topics(nmf, tfidf_feature_names, 10)




# lda = LatentDirichletAllocation(n_topics=8, max_iter=5,
#                                 learning_method='online', learning_offset=50.,
#                                 random_state=0)

model = lda.LDA(n_topics=8, n_iter=500, random_state=1)
model.fit(tfidf)

# doc = 0
# feature_index = tfidf[doc,:].nonzero()[1]
# tfidf_scores = zip(feature_index, [tfidf[doc, x] for x in feature_index])
#
# for w, s in [(tfidf_feature_names[i], s) for (i, s) in tfidf_scores]:
#   print (w, s)

'''
############################### Use lda package and CountVectorizer #########################################

vectorizer = CountVectorizer(max_features=600, analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
matrix =  vectorizer.fit_transform(select_line)
feature_names = vectorizer.get_feature_names()

vocab = feature_names
model = lda.LDA(n_topics=6, n_iter=500, random_state=1)
model.fit(matrix)
topic_word = model.topic_word_
n_top_words = 20

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_
print("type(doc_topic): {}".format(type(doc_topic)))
print("shape: {}".format(doc_topic.shape))

di = dict()
for n in range(50):
    topic_most_pr = doc_topic[n].argmax()
    if topic_most_pr in di :
        di[topic_most_pr].append(n)
    else:
        di[topic_most_pr] = [n]
    print("doc: {} topic: {}".format(n, topic_most_pr))



############################### Use genism package ################################################
doc_clean = [doc.split() for doc in select_line]
# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=50)

for top in ldamodel.print_topics(num_words=20):
  print(top)

# To print the train doc label:
for d in select_line:
    bow = dictionary.doc2bow(d.split())
    t = ldamodel.get_document_topics(bow)
    print(t)

# test
vec_bow = dictionary.doc2bow(select_line[0].lower().split())
lsi = models.LsiModel(doc_term_matrix, id2word=dictionary, num_topics=8)
vec_lsi = lsi[vec_bow]
print(vec_lsi)


############# workd cload test ####################
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS



for j in [0,1,2,3,4,5]:
    aa = ''.join(select_line[i] for i in di[j])
    wordcloud = WordCloud(
                      stopwords=STOPWORDS,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(aa)
    plt.subplot(4,2,j+1)
    plt.imshow(wordcloud)
    plt.show()
