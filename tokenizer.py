from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def remove_stopwords(words):
    return [word for word in words if word not in stopwords.words('english')]
    
def lemmatize(words):
    words = [WordNetLemmatizer().lemmatize(word, pos='n') for word in words]
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
    words = [WordNetLemmatizer().lemmatize(word, pos='a') for word in words]
    return words

def tokenize_twitter(text):
    return TweetTokenizer().tokenize(text)

def tokenize(text):
    return remove_stopwords(lemmatize(tokenize_twitter(text)))