"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: VI Contortium.
    ---------------------------------------------------------------------
	Description: This file is used to launch a minimal streamlit web
	application.

"""
# Streamlit Dependencies
import streamlit as st
import joblib,os
import base64
import pickle

# Loading Data Dependencies
import pandas as pd
import numpy as np
import nltk
import string
import re
import time

# Loading Data Processing Dependencies
import string
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, TreebankWordTokenizer

# Loading Model Building Dependencies
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import preprocessing

# Loading Model Evaluation Dependencies
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Loading Explorative  Data Analysis Dependencies
import re
import ast
import time
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# Display
sns.set(font_scale=1)
sns.set_style("white")

#download libraries
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Creating a sentiment_map and other variables
sentiment_map = {"Anti-Climate": -1, "Neutral": 0, "Pro-Climate": 1, "News-Fact": 2}
type_labels = raw.sentiment.unique()
df = raw.groupby("sentiment")
palette_color = sns.color_palette("dark")

scaler = preprocessing.MinMaxScaler()

# Data Preprocessing
# Dealing wiht Class Imbalances
# Resampling:
def resampling(df):
    """
        The functions takes in the dataframe and resample the classes based on class size.
        The class size is an average of the datasets among classes.
        This function resamples by downsampling classes with observations greater than the class size and
        upsampling the classes with observations smaller than the class size.
    """
    class_2 = df[df["sentiment"] == 2]
    class_1 = df[df["sentiment"] == 1]
    class_0 = df[df["sentiment"] == 0]
    class_n1 = df[df["sentiment"] == -1]
    class_size = 4265

    # Downsampling class_1 the PRO class
    rclass_1 = resample(class_1, replace=False, n_samples=class_size, random_state=42)
    #upsampling class 2 the NEWS class
    rclass_2 = resample(class_2, replace=True, n_samples=class_size, random_state=42)
    #upsampling class 0 NUETRAL class
    rclass_0 = resample(class_0, replace=True, n_samples=class_size, random_state=42)
    #upsampling class -1 the ANTI class
    rclass_n1 = resample(class_n1, replace=True, n_samples=class_size, random_state=42)
    sampled_df = pd.concat([rclass_2, rclass_1, rclass_0, rclass_n1])

    return sampled_df

# Resammpled train data
Resampled_raw_df = resampling(raw)

# Text Cleaning
# Removing Noise:
def cleaner(tweet):
    """
    This function takes in a dataframe and does the following:
    - convert letters to lowercase
    - remove URL links
    - remove # from hashtags
    - remove numbers
    - remove punctuation
    """
    tweet = tweet.lower()
    to_del = [
        r"@[\w]*",  # strip account mentions
        r"http(s?):\/\/.*\/\w*",  # strip URLs
        r"#\w*",  # strip hashtags
        r"\d+",  # delete numeric values
        r"U+FFFD",  # remove the "character note present" diamond
    ]
    for key in to_del:
        tweet = re.sub(key, "", tweet)

    # strip punctuation and special characters
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", " ", tweet)
    # strip excess white-space
    tweet = re.sub(r"\s\s+", " ", tweet)

    return tweet.lstrip(" ")

raw["message"] = raw["message"].apply(cleaner)

# Removing Stop Words:
stop_word = stopwords.words("english")
raw["message"] = raw["message"].apply(lambda x: " ".join([word for word in x.split() if word not in (stop_word)]))

# Tokenisation:
tokeniser = TreebankWordTokenizer()
raw["tokens"] = raw["message"].apply(tokeniser.tokenize)

# Lemmatisation:
def lemmas(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words]

lemmatizer = WordNetLemmatizer()
raw["lemma"] = raw["tokens"].apply(lemmas, args=(lemmatizer, ))

# Text Feature feature_extraction
# Split the features from the labels:
x = raw["message"]
y = raw["sentiment"]

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    st.set_page_config(page_title="GRIN Classifier", page_icon=":hash:", layout="centered")


	# Creates a main title and subheader on your page -
	# these are static across all pages
    st.markdown('---')
    st.title("GRIN Classifier")
    st.text("""The GRIN Tweet Classifier  analyses tweet sentiments and awareness of the general public towards
the impact of climate change. The App also evaluates the influence of demography
on the public's sentiments and awareness towards climate change. This can provide
organisations with valuable insight, which will enable them to tailor their
products and services towards the people in their demographic locations""")

    st.subheader("Classification Dataset")
    st.text("""Twitter is one of the top social media platforms, and one that encourages discussion
on socially relevent issues billions of people from a vast array of demographics. It is
for these reasons (amongst others) that twitter was deemed an appropriate data source
for this investigation.""")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
    options = ["Prediction", "Information","Exploratory Data Analysis", "About Team"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Prediction":
        st.subheader("Prediction")
    elif selection == "information":
        st.subheader("Information")
    elif selection == "EDA":
        st.subheader("Exploratory Data Analysis")
    else:
        st.subheader("About Team")

	# Building the About Team page
    if selection == "About Team":
        st.write("Meet the team behind the GRIN Classiffier")

        st.markdown(" ")

        John, Sandile, Hudson = st.columns(3)

        John.success("Lead, Digital Solutions")
        Sandile.success("Lead, Solution Architect")
        Hudson.success("Lead Data Scientist")

        with John:
            st.header("Meet John Lawal")
            st.image("resources/John Lawal.jpg")

        with Sandile:
            st.header("Meet Sandile Ngubane")
            st.image("resources/Sandile Ngubane.jpg")

        with Hudson:
            st.header("Meet Hudson Maina")
            st.image("resources/Hudson Maina.jpg")

        Duncan, Muhammad, Cosmus = st.columns(3)

        Duncan.success("Senior Data Engineer")
        Muhammad.success("Data Operations Solution Architect")
        Cosmus.success("Senior Data Scientist")

        with Duncan:
            st.header("Meet Duncan Okoth")
            st.image("resources/Duncan Okoth.jpg")

        with Muhammad:
            st.header("Meet Muhammad Yahya")
            st.image("resources/Muhammad Yahya.jpg")

        with Cosmus:
            st.header("Meet Cosmus Mutuku")
            st.image("resources/Cosmus Mutuku.jpg")


    # Building the Information page
    if selection == "Information":
        st.info("Brief Description")
        st.slider("Select a range of numbers", 0, 10)


        st.markdown("Some information here")
        st.markdown(" ")

        st.container()

        col1, col2 = st.columns(2)
        col1.success('1')
        col2.success('Important/most used words')

        with col1:
            st.checkbox('sentiment 1', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
            st.checkbox('sentiment 2', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
            st.checkbox('sentiment 3', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)
            st.checkbox('sentiment 4', value = False)#, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False)

        with col2:
            st.write('The word cloud function goes in here')

            st.markdown(" ")


        col3, col4 = st.columns(2)
        col3.success('Popular hashtags')
        col4.success('Mentions')

        with col3:
            st.write("List of popular hashtags function associated with sentiment goes in here")

        with col4:
            chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])

            st.bar_chart(chart_data)


        st.markdown("Some information here")
        st.subheader("Raw Twitter data and label")

        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")

        # Creating a text box for user input
        st.markdown("---")

        tweet_text = st.text_area("Enter Text","Type Here")

        st.markdown("---")

        st.write("The **GRIN App** employs **Logistic Regression** as the working model.")


        st.markdown("---")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            tweet_process = cleaner(tweet_text)
            vect_tweet = tweet_cv.transform([tweet_process])
      
            
	# 		# Load your .pkl file with the model of your choice + make predictions
	# 		# Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("Logistic_regression.pkl"),"rb"))
            prediction = predictor.predict(vect_tweet)
            
            
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))
            if prediction == 1:
                st.write(""" **The tweet supports the belief of man-made climate change.** """)

            elif prediction == 2:
                st.write(""" **The tweet link to factual news about climate change.** """)

            elif prediction == 0:
                st.write(""" **The tweet neither supports nor refutes the belief of man-made climate change.** """)

            else:
                st.write(""" **The tweet does not believe in man-made climate change** """)

        # Building the EDA Page
        if selection == "EDA":
            st.subheader("Tweet and Sentiment Exploration")
            hash_pick = st.checkbox("Hash-Tag")
            if hash_pick:
                val = st.selectbox("choose Tag Type", ["Hash-Tag", "Mentions"])
                sentiment_select = st.selectbox("Choose option", sentiment_map)
                iter_hash_select = st.slider("How many hash-tag", 1, 20, 10)
                if val == "Hash-Tag":
                    st.info("Popular Hast Tags")
                else:
                    st.info("Popular Mentions")
                valc = "hash_tag" if val == "Hash-Tag" else "mentions"
                result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
                col_type=valc)
                source = pd.DataFrame({
                "Frequency": result.values(),
                "Hash-Tag": result.keys()
                })
                val = np.array(list(result.values())).reshape(-1, 1)
                dd = (scaler.fit_transform(val)).reshape(1, -1)
                fig, ax = plt.subplots(1,2, figsize=(10, 15))
                ax[0].pie(data=source, x=result.values(), labels=result.keys(), colors=palette_color)#explode=dd[0], autopct='%.0f%%')
                word_cloud = WordCloud(background_color='white',
                                   width=512,
                                   height=384).generate(' '.join(result.keys()))
                ax[1].imshow(word_cloud)
                ax[1].axis("off")
                plt.show()
                st.pyplot(fig, use_container_width=True)

            word_pick = st.checkbox('Word Group(s)')
            if word_pick:
                st.info("Popular Group of Word(s)")
                sentiment_select_word = st.selectbox("Choose sentiment option", sentiment_map)
                word_amt = st.slider('Group of words', 1, 10, 5)
                group_amt = st.slider("Most frequent word groupings", 1, 10, 5)
                word_result = word_grouping(group_word_num=word_amt, ngram_iter_num=group_amt,
                                        sentiment_cat=sentiment_map[sentiment_select_word])
                st.table(pd.DataFrame({
                'Word group': word_result.keys(),
                'Frequency': word_result.values()
            }))

# Required to let Streamlit instantiate our web app.

if __name__ == '__main__':
    main()
