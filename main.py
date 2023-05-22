from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import nltk as nltk

#from nltk.tokenize import word_tokenize
#from nltk.tokenize import sent_tokenize

import re

df = pd.read_csv("emails.csv", encoding="latin-1")

def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)
def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without extra whitespaces
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()
def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    return len(text.split())

def remove_stopwords_func(text):
         stop = set(nltk.corpus.stopwords.words("english"))
         text_split= text.split(" ")
         return " ".join(t for t in text_split if t not in stop)

df['text'] = df['text'].apply(remove_stopwords_func)
df['Clean_text'] = df['text'].astype(str)
df['Clean_text'] = df['Clean_text'].str.lower()
df['Clean_text'] = df['Clean_text'].apply(remove_punctuation_func)
df['Clean_text'] = df['Clean_text'].apply(remove_extra_whitespaces_func)
df['textCount'] = df['text'].apply(word_count_func)
df['CleanCount'] = df['Clean_text'].apply(word_count_func)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("emails.csv", encoding="latin-1")
    df['text'] = df['text'].apply(remove_stopwords_func)
    df['Clean_text'] = df['text'].astype(str)
    df['Clean_text'] = df['Clean_text'].str.lower()
    df['Clean_text'] = df['Clean_text'].apply(remove_punctuation_func)
    df['Clean_text'] = df['Clean_text'].apply(remove_extra_whitespaces_func)
    df['textCount'] = df['text'].apply(word_count_func)
    df['CleanCount'] = df['Clean_text'].apply(word_count_func)    
    df.dropna(inplace=True)
    df['no_char'] = df['Clean_text'].apply(len)
    df['no_words'] = df['Clean_text'].apply(lambda x:len(nltk.tokenize.word_tokenize(x)))
    df['no_sent'] = df['Clean_text'].apply(lambda x:len(nltk.tokenize.sent_tokenize(x)))
    ham_corpus = []
    for mail in df[df['spam'] == 0]['Clean_text'].tolist():
        for word in mail.split():
            ham_corpus.append(word)
            
   
    X = df['text']
    y = df['spam']
    

    
    # Extract Feature With CountVectorizer
    CV = CountVectorizer()
    X = CV.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    

    from sklearn.neighbors import KNeighborsClassifier
 
        
    clf=KNeighborsClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

   
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = CV.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)