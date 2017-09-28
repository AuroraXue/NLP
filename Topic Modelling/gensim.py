import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# f = open('topics.txt', 'r')
# lines = f.readlines()
# f.close()
# start = False
# start_index =list()
# end_index = list()
# c = 0
#
# for i, line in enumerate(lines):
#     if line.startswith('<narr>'):
#         c=c+1
#         start = True
#         start_index.append(i)
#     if start and line.startswith('</top>'):
#         end_index.append(i)
#         start = False
#
# select_line = []
#
# for x,y in zip(start_index, end_index):
#     line = lines[(x+1):y]
#     line = ' '.join(line)
#     line = re.sub('[^a-zA-Z.]', ' ', line)
#     line = line.lower().split()
#     line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
#     line = ' '.join(line)
#     select_line.append(line)

# thefile = open('cleaned_sentence.txt', 'w')
# for line in select_line:
#     thefile.write("%s\n" % line)
# thefile.close()

################################ Part 1 Create necessary matrix #######################################
from gensim import corpora
f = open('cleaned_sentence.txt', 'r')
lines = f.readlines()
texts = []
for line in lines:
    line = re.sub('[^a-zA-Z]',' ', line)
    texts.append(line.split(" "))

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
print(dictionary)

print(dictionary.token2id)

# when you put in new sentence, how to fit the dictionary:
new_doc = "Women should prevent themselves from being horsewomen"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)

############################## Part 2 Transform Interface #############################################
import os
from gensim import corpora, models, similarities
if (os.path.exists("/tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

# step 1: Creating a transformation
#   The transformations are standard Python objects, typically initialized by means of a training corpus:

tfidf = models.TfidfModel(corpus)

# Try some new sentence: we already created above by using: new_vec = dictionary.doc2bow(new_doc.lower().split())
new_doc = "document said we should prevent being horsewomen"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(tfidf[new_vec])

# And we should apply tfidf on the whole corpus:
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

###################################### Part 3 Topic Modeling ############################################
# By using LSI
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(3,5)

def findTopic(ls):
    topic_score_ls = list()
    for i in ls:
        topic_score_ls.append(i[1])
        max_value = max(topic_score_ls)
        max_index = topic_score_ls.index(max_value)
    return max_index



for doc in corpus_lsi:
    print(doc, findTopic(doc))


# By using LDA

mm = corpora.MmCorpus('/tmp/deerwester.mm')
id2word = corpora.Dictionary.load('/tmp/deerwester.dict')
print(mm)

lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=8, update_every=1, chunksize=10000, passes=1)
lda.print_topics(8,5)

# If you want to try new sentence
new_doc = "document said we should prevent being horsewomen"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(tfidf[new_vec])

doc_lda = lda[new_vec] # this give out a list of probability of each topic in a tumple type
print(findTopic(doc_lda))

# for the trainning set:
corpus_lda = lda[corpus_tfidf]
for doc in corpus_lda:
    print(doc)