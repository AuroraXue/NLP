import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def get_top_models_NMF(file, n_topic, n_words):
    f = open(file, 'r')
    lines = f.readline()
    text_list = []
    for line in lines:
        line = re.sub('[^a-zA-Z.]', ' ', line)
        line = line.lower().split()
        line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
        line = ' '.join(line)
        text_list.append(line)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  # max_features=600,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(text_list)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Run NMF
    nmf = NMF(n_components=8, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    display_topics(nmf, tfidf_feature_names, 10)

