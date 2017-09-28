import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

example = "A relevant document may include one or more of the\
dietary intakes in the prevention of osteoporosis.\
Any discussion of the disturbance of nutrition and\
mineral metabolism that results in a decrease in \
bone mass is also relevant."
wordcloud = WordCloud(
                      stopwords=STOPWORDS,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(example)

plt.imshow(wordcloud)
plt.show()