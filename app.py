import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

pipe=pickle.load(open("Naive_model.pkl", "rb"))
st.title("Email/SMS Spam Classifier")
input_sms=st.text_area("Enter the message")
if st.button("Predict"):
    transformed_sms=text_transform(input_sms)
    output=pipe.predict([transformed_sms])[0]
    if output == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
