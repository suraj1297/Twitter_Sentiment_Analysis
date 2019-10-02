# Fisrt Things First
# Importing basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

# for nlp 
import re
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


# Importing Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

"""
Text is a highly unstructured form of data, various types of noise are present in it and the data 
is not readily analyzable without any pre-processing. The entire process of cleaning and 
standardization of text, making it noise-free and ready for analysis is known as text preprocessing.
 We will divide it into 2 parts:

1. Data Inspection
2. Data Cleaning

"""

# DATA INSPECTION
# Getting column names
train.columns
# Taking a look at top 10 tweets
train.tweet.head(10)

# Taking a look at racist tweets
train[train.label == 1]

# Taking a look at non racist tweets
train[train.label == 0]

# checking dimensions of the train and test dataset.
print(train.shape, test.shape)

# checking total of racist and non racist tweets
train["label"].value_counts()

""" In train dataset we have 29720(~7%) non racist tweets and 2242(~93%). Its clearly and imbalanced dataset """

# checking the distribution of length of the tweets, in terms of chars, in both train and test data.
length_train = train.tweet.apply(lambda x : len(x)) # train.tweet.str.len()
length_test = test.tweet.str.len()

# Plotting histogram using length
plt.hist(length_train, bins=20, color="red")
plt.hist(length_test, bins=20, color="blue")
plt.show()

# Checking max and min characterlength tweet in train
print(max(length_train), min(length_train))

# DATA CLEANING
"""
Before we begin cleaning, let’s first combine train and test datasets. Combining the datasets will 
make it convenient for us to preprocess the data. Later we will split it back into train and test data.
"""

data = train.append(test, ignore_index=True, sort=False)
data.shape

# contractions set for expanding contractions
contraction = { 
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"i'm":"i am",
"i've": "i have",
"i'd": "i had",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

# Saving stopwords
stopwords = set(STOPWORDS)
stopwords.remove('no') #removing negation from tweets as they contribute a lot
stopwords.remove('nor')
stopwords.remove('not')

# function for expanding contractions
def contractions(tweet):
    text = tweet.lower()
    words = []
    for word in text.split():
        if word in contraction:
            words.append(contraction.get(word))
        else:
            words.append(word)
    
    return " ".join(words)

#function for removing small words
def remove_small(tweet):
    
    words = []
    for word in tweet.split():
        if word in ["no","not","nor","never"]:
            words.append("not")
        elif len(word) > 2:
            words.append(word)
        
    return " ".join(words)


# Writing function for cleaning tweet
def clean(tweets):
    pre_processed_tweets = []
    
    for text in tweets:
        sentence = re.sub("@user", "", text) # removing user handles
        sentence = re.sub("[^a-zA-Z\'\s]","", sentence) # removing non characters
        sentence = re.sub("amp", "", sentence)  #removing word amp
        sentence = re.sub("bihday", "birthday", sentence)  #found bitrhday is missspelled a lot of times
        sentence = contractions(sentence) #expanding contractions
        sentence = re.sub(r"\'", "", sentence)
        sentence = " ".join(word for word in sentence.split() if word not in stopwords)
        sentence = remove_small(sentence) # removing short words
        stemmer = WordNetLemmatizer() # Stemming
        sentence = " ".join(stemmer.lemmatize(word, pos="v") for word in sentence.split())
        pre_processed_tweets.append(sentence.lower())
    return pre_processed_tweets

data["cleaned_tweets"] = clean(data.tweet)

# saving Cleaned data for future use

data.to_csv("cleaned_tweets", index=False)

# Using WordCloud to visualize the cleaned data
all_words = " ".join(text for text in data.cleaned_tweets)
word_frequncy = pd.Series(all_words.split()).value_counts().drop("not")
plt.figure(figsize=(12,10))    
wcd = WordCloud(width=800, height=500, background_color="black", random_state=21).generate_from_frequencies(word_frequncy)
plt.imshow(wcd, interpolation="bilinear")
plt.axis("off")
plt.show()

"""Well we can see the most dominating words in whole dataset's."""

"""
Well lets check the most dominating words in racist and non racist tweets
"""

# For racist tweets
racist_tweets = data.cleaned_tweets[data.label == 1]
racist_tweets_all_words = " ".join(text for text in racist_tweets)
racist_word_frequncy = pd.Series(racist_tweets_all_words.split()).value_counts().drop("not")
plt.figure(figsize=(12,10))    
wcd = WordCloud(width=800, height=500, background_color="black", random_state=21).generate(racist_tweets_all_words)
plt.imshow(wcd, interpolation="bilinear")
plt.axis("off")
plt.show()



# For non racist tweets
non_racist_tweets = data.cleaned_tweets[data.label == 0]
non_racist_tweets_all_words = " ".join(text for text in non_racist_tweets)
non_racist_word_frequncy = pd.Series(non_racist_tweets_all_words.split()).value_counts().drop("not")
plt.figure(figsize=(12,10))    
wcd = WordCloud(width=800, height=500, background_color="black", random_state=21).generate(non_racist_tweets_all_words)
plt.imshow(wcd, interpolation="bilinear")
plt.axis("off")
plt.show()

#Bar plot for non racist comments
plt.close()
plt.figure(figsize=(10,12))
sns.barplot(non_racist_word_frequncy.keys()[0:21],[word for word in dict(zip(non_racist_word_frequncy.keys(),non_racist_word_frequncy)).values()][0:21])
plt.xlabel("words")
plt.ylabel("frequency")
plt.title("Most words used in non racist tweets")
plt.show()

#Bar plot for most hateful/racist comments
top_words = [word for word in dict(zip(racist_word_frequncy.keys(),racist_word_frequncy)).keys()][0:21] # getiing top 20 words 
top_words_freq =[word for word in dict(zip(racist_word_frequncy.keys(),racist_word_frequncy)).values()][0:21]  # getting frequency of top 20 words.
plt.figure(figsize=(10,12))
sns.barplot(x=top_words,y=top_words_freq)
plt.xlabel("words")
plt.ylabel("frequency")
plt.title("Most words used in racist tweets")
plt.show()





## Featarization using BOW
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(min_df=10, max_features=1000)
bow = bow_vectorizer.fit_transform(data.cleaned_tweets)
bow.shape

## Featarization using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=10, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(data.cleaned_tweets)
tfidf.shape

## Word2Vec
from gensim.models import Word2Vec
tokenized_words = data.cleaned_tweets.apply(lambda x : x.split())
"""Step 1 : Instantiating model"""
w2v_model = Word2Vec(min_count=10,window=2,size=300,sample=6e-5,alpha=0.03,min_alpha=0.0007,negative=20)
"""STep 2 : Building Vocabulary"""
w2v_model.build_vocab(tokenized_words)
""" Trainig the model """
w2v_model.train(tokenized_words, total_examples=w2v_model.corpus_count, epochs=30)

""" Checking Similar words"""
w2v_model.wv.most_similar(positive=["trump"])
w2v_model.wv.most_similar(negative=["trump"])
w2v_model.wv.most_similar(positive=["love"], topn=6)

""" Checking Similarity between words"""
w2v_model.wv.similarity("trump", "obama")
w2v_model.wv.similarity("love", "trump")
w2v_model.wv.similarity("white", "racist")

""" Checking odd one out"""
w2v_model.wv.doesnt_match(["trump", "obama","racist"])
w2v_model.wv.doesnt_match(("love", "happy","trump"))
w2v_model.wv.doesnt_match(["trump", "white","black"])


"""Well we have created the vectors for each word now its time to create vector for each tweets/sentence"""
#creating function for creating sentence vector
def tweet_vec():
    tweet_vector = []
    """ LOOP for each tweet"""
    for tweet in tokenized_words:
        vector = np.zeros((300)).reshape(1,300)
        count = 0
        """ LOOP for each word in tweet"""
        for word in tweet:
            if word in w2v_model.wv.vocab:
                count +=1
                vector += w2v_model[word]
        if count:
            vector /=count
        tweet_vector.append(vector)
    return tweet_vector
''' Genearting list containing vectors'''
w2v_vectors =tweet_vec()


"""**************************************************************************************************"""
"""             CREATING DIFFERENT ML MODELS FOR PREDICTING        """



"""**************************************************************************************************"""
"""     1) Logistic Regression          """

    """ Bag of Words Features """
# Splitting my data into train and test
train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

# Splitting my data into train and validation data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_bow, train["label"], test_size=0.3, random_state=42)
""" Doing grid search for best model """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
regressor = LogisticRegression()
parameters = {"C":(0.01,0.1,1,3,4,5,10,11,12,13,15,17,18,20)}
clf = GridSearchCV(regressor, param_grid=parameters, cv=10, scoring= make_scorer(f1_score))
clf.fit(X_train,y_train)
""" Storing f1 scores"""
fscore_train = clf.cv_results_.get("mean_train_score")
fscore_test =clf.cv_results_.get("mean_test_score")
"""Plotting C vs f1 scores"""
plt.plot(list(parameters.get("C")),fscore_test, color = "blue", label="Test f1 score")
plt.plot(list(parameters.get("C")),fscore_train,color = "red", label="Train f1 score")
plt.xlabel("C")
plt.ylabel("F1Scores")
plt.legend()
plt.show()
""" From the plot we can conclude that after 4.5 or 7.5 the performance is stagnated hence can choose 4.5 as C"""
""" Instantiting the optimal logistic regressor """
optimal_regressor = LogisticRegression(C=4.5)
optimal_regressor.fit(X_train, y_train)
# Getiing f1 scores
probablity = optimal_regressor.predict_proba(X_valid)
y_pred = [1 if x>=0.3 else 0 for x in probablity[:,1]]

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)



    """ TFIDF Features """
# Splitting my data into train and test
train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_tfidf, train["label"], test_size=0.3, random_state=0)
""" Doing grid search for best model """
regressor = LogisticRegression()
parameters = {"C":(0.01,0.1,1,3,4,5,10,11,12,13,15,17,18,20)}
clf = GridSearchCV(regressor, param_grid=parameters, cv=10, scoring= make_scorer(f1_score))
clf.fit(X_train,y_train)
""" Storing f1 scores"""
fscore_train = clf.cv_results_.get("mean_train_score")
fscore_test =clf.cv_results_.get("mean_test_score")
"""Plotting C vs f1 scores"""
plt.plot(list(parameters.get("C")),fscore_test, color = "blue", label="Test f1 score")
plt.plot(list(parameters.get("C")),fscore_train,color = "red", label="Train f1 score")
plt.xlabel("C")
plt.ylabel("F1Scores")
plt.legend()
plt.show()
""" From the plot we can conclude that after 4.5 or 7.5 the performance is stagnated hence can choose 4.5 as C"""
""" Instantiting the optimal logistic regressor """
optimal_regressor = LogisticRegression(C=4.1)
optimal_regressor.fit(X_train, y_train)
# Getiing f1 scores
probablity = optimal_regressor.predict_proba(X_valid)
y_pred = [1 if x>=0.3 else 0 for x in probablity[:,1]]

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)



    """ Word2Vec Features """
# Splitting my data into train and test
w2v = np.array(w2v_vectors).reshape(-1,300)
train_w2v = w2v[:31962,:]
test_w2v = w2v[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_w2v, train["label"], test_size=0.3, random_state=0)
""" Doing grid search for best model """
regressor = LogisticRegression()
parameters = {"C":(0.01,0.1,1,3,4,5,10,11,12,13,15,17,18,20)}
clf = GridSearchCV(regressor, param_grid=parameters, cv=10, scoring= make_scorer(f1_score))
clf.fit(X_train,y_train)
""" Storing f1 scores"""
fscore_train = clf.cv_results_.get("mean_train_score")
fscore_test =clf.cv_results_.get("mean_test_score")
"""Plotting C vs f1 scores"""
plt.plot(list(parameters.get("C")),fscore_test, color = "blue", label="Test f1 score")
plt.plot(list(parameters.get("C")),fscore_train,color = "red", label="Train f1 score")
plt.xlabel("C")
plt.ylabel("F1Scores")
plt.legend()
plt.show()
""" From the plot we can conclude that after 4.5 or 7.5 the performance is stagnated hence can choose 4.5 as C"""
""" Instantiting the optimal logistic regressor """
optimal_regressor = LogisticRegression(C=17)
optimal_regressor.fit(X_train, y_train)
# Getiing f1 scores
probablity = optimal_regressor.predict_proba(X_valid)
y_pred = [1 if x>=0.3 else 0 for x in probablity[:,1]]

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)



"""**************************************************************************************************"""
"""     2) SVM         """

    """ Bag of Words Features """
# Splitting my data into train and test
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_bow, train["label"], test_size=0.3, random_state=0)
""" Doing grid search for best model """
from sklearn.svm import SVC
regressor = SVC(kernel="linear")
tuned_parameters = {'C': [1, 10, 100]} #{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
clf = GridSearchCV(regressor, param_grid=tuned_parameters, cv=10, scoring= make_scorer(f1_score))
clf.fit(X_train,y_train)
""" Printing Results"""
clf.cv_results_

""" Storing f1 scores"""
fscore_train = clf.cv_results_.get("mean_train_score")
fscore_test =clf.cv_results_.get("mean_test_score")
"""Plotting C vs f1 scores"""
plt.plot(list(parameters.get("C")),fscore_test, color = "blue", label="Test f1 score")
plt.plot(list(parameters.get("C")),fscore_train,color = "red", label="Train f1 score")
plt.xlabel("C")
plt.ylabel("F1Scores")
plt.legend()
plt.show()
""" From the plot we can conclude that after 4.5 or 7.5 the performance is stagnated hence can choose 4.5 as C"""
""" Instantiting the optimal svc classifier """
optimal_regressor = SVC(kernel="linear",C=17, probability=True)
optimal_regressor.fit(X_train, y_train)
# Getiing f1 scores
probablity = optimal_regressor.predict_proba(X_valid)
y_pred = [1 if x>=0.3 else 0 for x in probablity[:,1]]

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)



"""**************************************************************************************************"""
"""     3) RandomForest         """

    """ Bag of Words Features """
# Splitting my data into train and test
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_bow, train["label"], test_size=0.3, random_state=0)

""" Instantiting the  randonforest classifier """
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=400,random_state=11)
classifier.fit(X_train, y_train)
# Getiing f1 scores
y_pred = classifier.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)


    """ tfidf Features """
# Splitting my data into train and test
train_idf = tfidf[:31962,:]
test_idf = tfidf[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_idf, train["label"], test_size=0.3, random_state=0)

""" Instantiting the  randonforest classifier """
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500,random_state=11)
classifier.fit(X_train, y_train)
# Getiing f1 scores
y_pred = classifier.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)


    """ Word2Vec Features """
# Splitting my data into train and test
w2v = np.array(w2v_vectors).reshape(-1,300)
train_w2v = w2v[:31962,:]
test_w2v = w2v[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_w2v, train["label"], test_size=0.3, random_state=0)

""" Instantiting the  randonforest classifier """
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=400,random_state=11)
classifier.fit(X_train, y_train)
# Getiing f1 scores
y_pred = classifier.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)


"""**************************************************************************************************"""
"""     4) XGBoost         """

    """ Bag of Words Features """
# Splitting my data into train and test
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_bow, train["label"], test_size=0.3, random_state=0)

""" Instantiting the  xgboost classifier """
from xgboost import XGBClassifier as xgb
classifier = xgb(n_estimators=2000, max_depth=6)
classifier.fit(X_train, y_train)
# Getiing f1 scores
y_pred = classifier.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)

    """ TFIDF """
# Splitting my data into train and test
train_idf = tfidf[:31962,:]
test_idf = tfidf[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_idf, train["label"], test_size=0.3, random_state=0)

""" Instantiting the  xgboost classifier """
classifier = xgb(n_estimators=1000, max_depth=6)
classifier.fit(X_train, y_train)
# Getiing f1 scores
y_pred = classifier.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)

    """ Word2Vec Features """
# Splitting my data into train and test
w2v = np.array(w2v_vectors).reshape(-1,300)
train_w2v = w2v[:31962,:]
test_w2v = w2v[31962:,:]

# Splitting my data into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(train_w2v, train["label"], test_size=0.3, random_state=0)

""" Instantiting the  xgboost classifier """
classifier = xgb(n_estimators=1000, max_depth=6)
classifier.fit(X_train, y_train)
# Getiing f1 scores
y_pred = classifier.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)

""" Hence until now the best score we got was from xgboost """

""" Lets tune hyperparameters of  xgboost """
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train) 
dvalid = xgb.DMatrix(X_valid, label=y_valid)
# Parameters that we are going to tune 
params = {
    'objective':'binary:logistic',
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1
 }

#We will prepare a custom evaluation metric to calculate F1 score.

def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds >= 0.3).astype(np.int)
    return [('f1_score', f1_score(labels, preds))]

"""
General Approach for Parameter Tuning

We will follow the steps below to tune the parameters.

Choose a relatively high learning rate. Usually a learning rate of 0.3 is used at this stage.

Tune tree-specific parameters such as max_depth, min_child_weight, subsample, colsample_bytree keeping the learning rate fixed.

Tune the learning rate.

Finally tune gamma to avoid overfitting."""

#Tuning max_depth and min_child_weight

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6,10)
     for min_child_weight in range(5,8)
 ]

max_f1 = 0. # initializing with 0 
best_params = None 

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
     # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

     # Cross-validation
    cv_results = xgb.cv(params, dtrain, feval= custom_eval, num_boost_round=200,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=10
    )     

    # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    
    boost_rounds = cv_results['test-f1_score-mean'].argmax()    
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))    
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (max_depth,min_child_weight) 

print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))


#Updating max_depth and min_child_weight parameters.
params['max_depth'] = 9 
params['min_child_weight'] = 5

#Tuning subsample and colsample

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(5,10)]
    for colsample in [i/10. for i in range(5,10)] ]
max_f1 = 0. 
best_params = None 

for subsample, colsample in gridsearch_params:
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
     # Update our parameters
    params['colsample'] = colsample
    params['subsample'] = subsample
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=200,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=10
    )
     # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (subsample, colsample) 

print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))



#Updating subsample and colsample_bytree
params['subsample'] = .8 
params['colsample_bytree'] = .5

max_f1 = 0. 
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
     # Update ETA
    params['eta'] = eta

     # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=1000,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=20
    )

     # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = eta 
print("Best params: {}, F1 Score: {}".format(best_params, max_f1))

params['eta'] = best_params

#Let’s have a look at the final list of tuned parameters.
print(params)


#Finally we can now use these tuned parameters in our xgboost model.Early stopping of 10 which means if the
#model’s performance doesn’t improve under 10 rounds, then the model training will be stopped.

""" Way1- To train final model"""
# Training/Learning final Model
final_xgb_model = xgb.train(
    params,
    dtrain,
    feval= custom_eval,
    num_boost_round= 1000,
    maximize=True,
    evals=[(dvalid, "Validation")],
    early_stopping_rounds=10
 )
""" Way2- To train final model"""
X_train, X_valid, y_train, y_valid = train_test_split(train_w2v, train["label"], test_size=0.3, random_state=0)
final = xgb.XGBClassifier(**params,n_estimators=1000)
final.fit(X_train,y_train)
# Getiing f1 scores
y_pred = final.predict(X_valid)

f1Score = f1_score(y_valid, y_pred)
print(f1Score*100)

""" After different evaluation of different models we choose xgboost as our go to model and after hypertuning
its paramters and training it using the best parameters the we achieved the best evaluation score that we can achieve.
"""

"""" Hence our model is ready names as xgb_final_model"""
