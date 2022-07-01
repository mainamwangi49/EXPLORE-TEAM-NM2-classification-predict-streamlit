# Streamlit Dependencies
import streamlit as st
import joblib,os

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

# Prediction outputs
import emoji

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

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Creating a sentiment_map and other variables
sentiment_map = {"Anti-Climate": -1, "Neutral": 0, "Pro-Climate": 1, "News-Fact": 2}
type_labels = raw.sentiment.unique()
df = raw.groupby("sentiment")
palette_color = sns.color_palette("dark")

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
            st.image("https://static.streamlit.io/examples/cat.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with Sandile:
            st.header("Meet Sandile Ngubane")
            st.image("https://static.streamlit.io/examples/dog.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with Hudson:
            st.header("Meet Hudson Maina")
            st.image("https://static.streamlit.io/examples/owl.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        Duncan, Muhammad, Cosmus = st.columns(3)

        Duncan.success("Senior Data Engineer")
        Muhammad.success("Data Operations Solution Architect")
        Cosmus.success("Senior Data Scientist")

        with Duncan:
            st.header("Meet Duncan Okoth")
            st.image("https://static.streamlit.io/examples/cat.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with Muhammad:
            st.header("Meet Muhammad Yahya")
            st.image("https://static.streamlit.io/examples/dog.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

        with Cosmus:
            st.header("Meet Cosmus Mutuku")
            st.image("https://static.streamlit.io/examples/owl.jpg")

            with st.expander("Brief Bio"):
                st.write("""Bio goes in here""")

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

        option = st.selectbox("Please select your model",(
            "LogisticRegression", "KNeighborsClassifier",
            "SVC",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "LinearSVC",
            "MultinomialNB"))

        st.write("You selected:", option)

        if option == "LogisticRegression":
            with st.expander("See explaination"):
                st.write("""The logistic regression classifier is a supervised machine learning
                classification algorithm that is used to predict the probability of a categorical
                dependent variable. In this method, the data is fitted to a logit function and
                labelled as the class for which it has the highest probability of belonging to.""")

        elif option == "KNeighborsClassifier":
            with st.expander("See explaination"):
                st.write("""The K-nearest neighbors (KNN) algorithm works by finding the distances
                between a query and all the examples in the data, selecting the specified number of
                examples (K) closest to the query, and voting for the most frequent label.""")

        elif option == "SVC":
            with st.expander("See explaination"):
                st.write("""The Support Vector Machine (SVM) Classifier is a discriminative classifier
                formally defined by a separating hyperplane. When labelled training data is passed to
                the model, also known as supervised learning, the algorithm outputs an optimal hyperplane
                which categorizes new data. In the SVM algorithm, we plot each data item as a point in
                n-dimensional space (where n is a number of features you have) with the value of each
                feature being the value of a particular coordinate. Then, we perform classification by
                finding the hyper-plane that differentiates the two classes very well.""")

        elif option == "DecisionTreeClassifier":
            with st.expander("See explaination"):
                st.write("""A decision tree classifier builds a model from the top down in a tree structure,
                where each decision point is called a 'Decision Node' and the end point is called s 'Leaf Node'.
                The data is broken down into smaller and smaller subsets while a associated decision tree is
                progressively built to capture the decisions that determine the paths of the tree.""")

        elif option == "RandomForestClassifier":
            with st.expander("See explaination"):
                st.write("""A Random forest creates decision trees on randomly selected data samples,
                gets prediction from each tree and selects the best solution by means of voting.""")

        elif option == "LinearSVC":
            with st.expander("See explaination"):
                st.write("""The objective of a Linear Support Vector Classifier is to return a "best fit" hyperplane
                that categorises the data. It is similar to SVC with the kernel parameter set to ’linear’,
                but it is implemented in terms of liblinear rather than libsvm, so it has more flexibility in the
                choice of penalties and loss functions and can scale better to large numbers of samples. The Linear SVC is also very similar to Logistic Regression (LR). It differs mostly in applying a buffer
                margin (determined by the support vectors - vectors that lie on the buffer margins) and using 'hinge loss'
                as apposed to log loss by LR. The Linear SVC creates a separating hyperplane that optimally separates
                the different classes ('hinge loss' is the function that determines the buffer zone and hence the separation).
                In simpler terms, the Linear SVC allows for a margin of error when determining how to optimally separate
                the different classes, making it more robust to data with classes that aren't very clearly separated.""")

        else:
            with st.expander("See explaination"):
                st.write("""The Multinomial Naive Bayes model estimates the conditional probability of a particular feature
                given a class and uses a multinomial distribution for each of the features. The model assumes that each
                feature makes an independent and equal contribution to the outcome.""")


        st.markdown("---")

        if st.button("Classify"):

            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()

            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Support_Vector_Machine_C.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

if __name__ == '__main__':
# Required to let Streamlit instantiate our web app.
	main()
