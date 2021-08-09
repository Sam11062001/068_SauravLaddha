#from google.colab import drive
#drive.mount("/content/drive")

# importing modules
import nltk
import matplotlib.pyplot as plt
import pandas as pd

# Performing the Raw Text Analysis
random_text = """“When you speak the language, it comes with cultural understanding that's more important than language itself; understanding how these countries work, ways of doing business, handling meetings..."
https://danielsjross.substack.com/p/coming-out-of-the-shadows-fintech
#Mexico #Colombia #Brasil 
@fintech_io https://twitter.com/danielsjross/status/1405472708624781314"""




# importing modules
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

removed_link_hash_text = re.sub(r'https?:\/\/.*[\r\n]*', '', random_text)
removed_link_hash_text = re.sub(r'#', '', removed_link_hash_text)
print(removed_link_hash_text)

print('\033[92m' + random_text)
print('\033[92m' + removed_link_hash_text)





from nltk.tokenize import sent_tokenize
# downloading punkt from nltk
nltk.download("punkt")

text="""Expectations of bad weather, which would make the roads impassable, had driven the relief forces forward.
Explanations of such persistently poor performance included multiple repair and maintenance problems affecting both agricultural and industrial operations, poor management and bad weather."""

tokenized_text=sent_tokenize(text)
print(tokenized_text)




# breaking paragraph into the words words
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

# performing the frequency distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
fdist.most_common(4)

# showing the Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30, cumulative = False, color = "green")
plt.show()

# showing the stop words
from nltk.corpus import stopwords
# downloading stopwords from nltk
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
print(stop_words)

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

# showing how to perform stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

# Lexicon Normalization
# performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))

