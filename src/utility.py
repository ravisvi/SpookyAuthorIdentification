from nltk.corpus import stopwords
import string
def clean_text(text):
    # remove punctuations
    cleaned_text = text.translate(None, string.punctuation)
    
    # upper to lower
    cleaned_text = [word.lower() for word in cleaned_text.split(' ')]

    # removing stopwords
    return ' '.join([word for word in cleaned_text if word not in stopwords.words('english')])