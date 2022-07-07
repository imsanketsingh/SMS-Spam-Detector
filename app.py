import streamlit as slt  #streamlit us used to create a web app so that we can host it and use our work anywhere anytime
                         # reference: https://docs.streamlit.io/
import pickle
import string
import nltk
from nltk.corpus import stopwords



def transform_smsTxt(text):
    text= text.lower()  # changes everything in smallcase to avoid duplication/misclassification
    text= nltk.word_tokenize(text)  # to make a list of words of the text
    #now text is a list
    y= []
    
    for i in text:   # to remove the special characters beacuse they don't carry much info about the sms being span or not
        if(i.isalnum()):
            y.append(i)
    # now we ll remove the stop words (the words that doesn't make sense but used in sentence formation(e.g are, is, has, etc.))
    # check stopwords here
#     from nltk.corpus import stopwords
#     stopwords.words('english')
    
    # similarly we wanna remove punctuations also
    # check punctuations here
#     import string 
#     string.punctuation
    
    # Now we will do stemming i.e. we ll convert the words that have same sense(for e.g. walk, walked, walks, walking) into their root word
    from nltk.stem.porter import PorterStemmer
    pts = PorterStemmer()
    
    text=y.copy()
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y.copy()
    y.clear()
    for i in text:
        y.append(pts.stem(i))
    
    return " ".join(y) # to finally convert into string

tfidf= pickle.load(open('vectorizer.pkl', 'rb'))
model= pickle.load(open('model.pkl', 'rb'))

slt.title("SMS Spam Detector")
input_sms= slt.text_area("Enter the SMS")

if(slt.button('Predict')):
    transformed_sms= transform_smsTxt(input_sms)
    vectorized_input= tfidf.transform([transformed_sms])
    output= model.predict(vectorized_input)[0]

    if(output==1):
        slt.header("Spam")
    else:
        slt.header("Not Spam")


slt.write("""
Project by **Sanket** \n
Source Code: [GitHub](https://github.com/imsanketsingh)\n
Connect with me: [LinkedIn](https://www.linkedin.com/in/sanket-kumar-singh-b698191b8/)
""")