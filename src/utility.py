from nltk.corpus import stopwords
import string
def clean_text(text):
    # upper to lower
    cleaned_text = [word.lower() for word in cleaned_text.split(' ')]
    
    # remove punctuations
    cleaned_text = text.translate(None, string.punctuation)

    # removing stopwords
    return ' '.join([word for word in cleaned_text if word not in stopwords.words('english')])