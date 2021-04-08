import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import string
import re #for text process
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

img1 = Image.open('./data/Hotel_review/like-dislike-featured.jpg')
img2 = Image.open('./data/Hotel_review/rating-star.jpg')

img1 = img1.resize((300,300))
img2 = img2.resize((300,300))

st.header('Hotel review sentiment analysis')
st.image('./data/Hotel_review/review.jpg')
st.write("The customer analysis would be useful when you get the reviews for many customer\
          in the text form. Usually, The customer give you the rating star or like and diskike\
          in the Hotel's webpage ")

st.write('Otherwise, the real world information come from the many type such as the email , \
    comment box and direct message')


# st.beta_container()
# col1,col2 = st.beta_columns(2)
# with col1:
#     col1 = st.image(img1)
# with col2:
#     col2 = st.image(img2)

df = pd.read_csv('./data/Hotel_review/hotel_review_cleaned.csv')

# st.sidebar.radio('show comment',[1,2])

# def pre_process_dataframe(df):
#     dummie = pd.get_dummies(df.Is_Response,prefix='res')
#     df_label = pd.concat([df,dummie],axis=1)
#     df = df_label.drop(columns=['Is_Response','res_not happy'])
#     return df

def text_process(text):
#   word = text.lower()
#   word = word.split()
    word = re.sub('[^A-za-z]',' ',text)
    word = word.lower()
    word = word.split()
    word = [words for words in word if words not in stopwords.words('english')]
    word = [WordNetLemmatizer().lemmatize(words) for words in word]
    word = ' '.join(word)
    return word

classifier = pickle.load(open('./data/Hotel_review/hotel_review.pkl','rb'))
cv = CountVectorizer()
cv.fit(df['review'])

raw_text = st.text_input('Please leave your comment about my Hotel')
comment = text_process(raw_text)
comment = cv.transform([comment])

predict_ans = classifier.predict(comment)

while len(raw_text) > 10:
    if predict_ans == 0:
        st.write('Sorry for inconvenience.')
        break
    elif predict_ans == 1:
        st.write('Thank you for overwarming support my hotel')
        break


