# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# 0. 개요
# 1. Library & Data Import
# 2. 데이터셋 살펴보기
# 3. 텍스트 전처리
#     1. 정규표현식 적용
#     2. 말뭉치 만들기
#     3. 토큰화
#     4. Word count 
#         - Bag of words
#         - TF-IDF 변환
# 4. 감성 분석
#     1. 데이터셋 생성
#         - Label
#         - Feature
#     2. Training set / Test set 나누기
#     3. Logistic regression 모델 학습
# 5. 긍정/부정 키워드 분석
# 6. 호텔리뷰 감성 (긍정/부정) 예측함수
#     1. Logistic regression
#     2. SVM
#     3. Decision tree
#     4. Random forest
#     5. KNN
#     6. LSTM
#     7. Grid search
#     8. 각 모델별 AUC score 비교
# 7. 개선점

# # 0. 개요

# https://hyemin-kim.github.io/2020/08/29/E-Python-TextMining-2/#5-%EA%B8%8D%EC%A0%95-%EB%B6%80%EC%A0%95-%ED%82%A4%EC%9B%8C%EB%93%9C-%EB%B6%84%EC%84%9D

# 호텔의 리뷰 데이터(평가 점수 + 평가 내용)을 활용해 다음 2가지 분석을 진행한다.
#
# 1. 리뷰 속에 담긴 사람의 긍정 / 부정 감성을 파악하여 분류할 수 있는 감성 분류 예측 모델을 만든다.
#
# 2. 만든 모델을 활용해 긍정 / 부정 키워드를 출력해, 이용객들이 느낀 호텔의 장,단점을 파악한다.

# # 1. Library & Data Import

# +
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# -

df =pd.read_csv("C:/Users/gilon/data/hotel_review.csv")

df

# # 2. 데이터셋 살펴보기

df.head()

df.shape

# 결측치
df.isnull().sum()

# information
df.info()

# content 확인 : 첫번째 리뷰 확인
df['content'][0]

# content 확인 : 101번째 리뷰 확인
df['content'][100]

del df['Unnamed: 0']

df

#본문의 제목과 내용 합치기
df['review']=df['Title']+" "+df['content']

df

df = df.drop(['Title', 'content'], axis=1)

df

# Text Mining을 적용할 필요가 없는 문자들은 정규표현식을 이용해서 제거한다.
#
# 1. 개행 문자(\n) 제거
# 2. 특수문자(! ? , .) 제거
# 3. 이모티콘 제거 -> 이모티콘도 감정을 표현하는 하나의 방식이기 때문에 텍스트로 대치해주는 방식을 적용하면 모델의 성능을 올릴 수 있을 것으로 기대됨 
#     - 예시: (😍-> (하트) / 👍🏻-> (굿) )
# 4. 띄어쓰기, 맞춤법 검사

# # 3. 텍스트 전처리

# <텍스트로 이루어진 데이터를 분석하기 위해서 무엇을 해야 할까?>
# 1. 전처리 작업 : Tokenization(문장을 단어로 쪼개기), 불용어 제거, 단어 정규화(ex. apples → apple) 등의 전처리 작업
# 2. 컴퓨터가 이해할 수 있는 데이터로 변환 : 문자를 숫자로 변환하는 작업을 수행 -> BoW(Bag of Words)

# ## 3-1. 정규 표현식 적용 
# 영어가 아닌 문자는 제거하며 단어만 남기도록 한다.

import re

# 1. 영문자 이외 문자는 공백으로 변환한다.

only_english = re.sub('[^a-zA-Z]', ' ', df['review'][0])

df['review'][0]

# 2. 대문자는 소문자로 변환
#
# 영어의 경우 문장의 시작이나 고유명사는 대문자로 시작하여 분석할때 "Apple"과 "apple"을 서로 다른 단어로 취급하게 된다. 따라서 모든 단어를 소문자로 변환한다.

# 소문자 변환
no_capitals = only_english.lower().split()

# 3. 불용어 제거
#
#     * 불용어 : 학습 모델에서 예측이나 학습에 실제로 기여하지 않는 텍스트
#
# I, that, is, the, a  등과 같이 자주 등장하는 단어이지만 실제로 의미를 찾는데 기여하지 않는 단어들을 제거하는 작업이 필요하다.

from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# 불용어 제거
stops = set(stopwords.words('english'))
no_stops = [word for word in no_capitals if not word in stops]

# 4. 어간 추출
#
#     * see, saw, seen
#     * run, running, ran
#
# 위 예시처럼 어형이 과거형이든 미래형이든 하나의 단어로 취급하기 위한 처리작업이다.
#
# nltk에서 제공하는 형태소 분석기를 사용하는데 여러가지 어간추출 알고리즘(Porter, Lancaster, Snowball 등등)이 존재한다.

# 어간 추출
stemmer = nltk.stem.SnowballStemmer('english')
stemmer_words = [stemmer.stem(word) for word in no_stops]


# **전체 코드**

def data_text_cleaning(data):
 
    # 영문자 이외 문자는 공백으로 변환
    only_english = re.sub('[^a-zA-Z]', ' ', data)
 
    # 소문자 변환
    no_capitals = only_english.lower().split()
 
    # 불용어 제거
    stops = set(stopwords.words('english'))
    no_stops = [word for word in no_capitals if not word in stops]
 
    # 어간 추출
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmer_words = [stemmer.stem(word) for word in no_stops]
 
    # 공백으로 구분된 문자열로 결합하여 결과 반환
    return ' '.join(stemmer_words)


# 첫번째 리뷰
df['review'][0]

# 첫번째 리뷰에 대한 정규 표현식 함수(data_text_cleaning) 적용
data_text_cleaning(df['review'][0])

# ## 3-2. 말뭉치 만들기

# 말뭉치 생성
words = "".join(df['review'].tolist())
words

# 전체 말뭉치(corpus)에 정규 표현식 함수(data_text_cleaning) 적용

corpus = data_text_cleaning(words)
corpus

# ## 3-3. 토큰화 
#
# 문자열에서 단어로 분리시킨다.

from nltk.tokenize import word_tokenize

word_token = word_tokenize(corpus)
word_token

# 해당 리뷰 데이터에 해당하는 전체 단어의 개수
len(word_token) 

# **각 단어들의 빈도 탐색**

from collections import Counter

counter = Counter(word_token)

counter.most_common(10)

# ## 3-4.  Word Count

# **1. BoW 벡터 생성**
#
# https://doitgrow.com/15

# 데이터 형태를 통일하기 위해 전체 문서(또는 문장)를 보고 단어 가방(BoW)를 설계/제작한다.

len(word_token)

# 전체 리뷰를 통해 89647개의 고유 단어들이 있는 것을 확인하였다. 

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(df['review'].tolist()) # 가지고 있는 문장들로 가방을 설계
vectors = cv.transform(df['review'].tolist()).toarray() # 단어들을 가방에 정리하여 넣음

print(vectors)

vectors.shape

# **2. TF-IDF(Term Frequency-Inverse Document Frequency) 적용**

# BoW의 한계점과 이를 보완하기 위해서 Bag of Words 벡터에 대해서 TF-IDF변환을 진행한다.
# * 텍스트 정보를 BoW를 통해 언어 모델로 해석하려고 한다면 몇 가지 문제점이 존재한다.
#
# 1) 불용어(의미 없는 단어)를 제대로 제거하지 못하면 원하지 않는 편향된(biased) 결과가 얻어질 수 있다. -> TF-IDF 모델로 해결
#
# 2) 문장(또는 문서)의 의미가 단어 순서에 따라 달라질 수 있지만 BoW 모델은 이를 반영할 수 없다. -> TF-IDF 모델로도 해결 불가능, 이를 해결하기 위한 방법으로 n-gram을 사용하기도 하지만 n-gram을 사용하면 벡터의 차원이 기하급수적으로 늘어나거나, 벡터가 희소(Sparse) 형태로 표현될 가능성이 높아져서 성능에 영향을 미칠 수 있다.

# **TF-IDF 방법**
#
# 문서를 특징짓는 중요한 단어는 너무 적게 나오지도 너무 많이 나오지도 않는다. 따라서 너무 많이 출현했거나 너무 적게 출현한 단어들에는 패널티를 부여하고, 적절하게 출현한 단어들에는 가중치를 부여하여 문서의 특징을 더 많이 대변하도록 만들어 준다.

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(vectors)

print(tf_idf_vect.shape)

# * 한 행(row)은 한 리뷰를 의미 : 2889
# * 한 열(column)은 한 단어를 의미 : 7282

# 첫 번째 리뷰에서의 단어 중요도(TF-IDF 값) -- 0이 아닌 것만 출력
print(tf_idf_vect[0])

# 첫 번째 리뷰에서 모든 단어의 중요도 -- 0인 값까지 포함
print(tf_idf_vect[0].toarray().shape)
print(tf_idf_vect[0].toarray())

cv.vocabulary_

# # 4. 감성 분석

# 감성 분석 예측 모델 : 이용자 리뷰의 평가 내용을 통해 이 리뷰가 긍정적인지 또는 부정적인지를 예측하여, 이용자의 감성을 파악한다.
# * x값 (feature 값) = 이용자 리뷰의 평가 내용
# * y값 (label 값) = 이용자의 긍/부정 감성

# ## 4-1. 데이터셋 생성

# **1. Label**

# 이용자의 감성을 대표할 수 있는“score”변수는 1 ~ 5의 value를 가진다. 따라서 "score" 변수 (rating: 1 ~ 5)를 이진 변수 (긍정: 1, 부정:0)으로 변환해야 한다.

df

df['review'][566]

df.sample(10) # 무작위로 10개의 샘플을 추출

# 리뷰 내용와 평점을 살펴보면, 4 ~ 5점 리뷰는 대부분 긍정적이었지만, 1 ~ 3점 리뷰에서는 부정적인 평가가 대부분이었다.
# 따라서 4점, 5점인 리뷰는 "긍정적인 리뷰"로 분류하여 1을 부여하고, 1 ~ 3점 리뷰는 "부정적인 리뷰"로 분류하여 0을 부여하도록 한다.

df['score'].hist()


# +
def rating_to_label(rating):
    if rating > 3:
        return 1
    else:
        return 0
    
df['y'] = df['score'].apply(lambda x: rating_to_label(x))
# -

df.head()

df["y"].value_counts()

# **2. Feature**

# 모델의 Feature 변수는 리뷰에서 추출된 형태소와 그들의 중요도를 나타나는 tf_idf_vect로 대체한다.

# ## 4-2. Training set / Test set 나누기

from sklearn.model_selection import train_test_split

x = tf_idf_vect
y = df['y']

x.shape

y.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

x_train.shape, y_train.shape

x_test.shape, y_test.shape

# ## 4-3. 모델 학습

# ### 1. Logistic Regression 모델 학습

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# fit in training set
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

# predict in test set
y_pred = lr.predict(x_test)

# 분류 결과 평가

# +
# classification result for test set

print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('precision: %.2f' % precision_score(y_test, y_pred))
print('recall: %.2f' % recall_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))

# +
# confusion matrix

from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
plt.title('Confusion Matrix')
plt.show()
# -

# 모델 평가결과를 살펴보면, 모델이 지나치게 긍정(“1”)으로만 예측하는 경향이 있다. 따라서 긍정 리뷰는 잘 예측하지만, 부정 리뷰에 대한 예측 정확도가 매우 낮다. 이는 샘플데이터의 클래스 불균형으로 인한 문제로 보이므로 클래스 불균형 조정을 진행한다.

# ### 2. 샘플링 재조정

df['y'].value_counts()

positive_random_idx = df[df['y']==1].sample(819, random_state=1).index.tolist()
negative_random_idx = df[df['y']==0].sample(819, random_state=1).index.tolist()
# random_state : random 함수의 seed값 -> random 값을 고정

random_idx = positive_random_idx + negative_random_idx
x = tf_idf_vect[random_idx]
y = df['y'][random_idx]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)

x_train.shape, y_train.shape

x_test.shape, y_test.shape

# ### 3. Logistic Regression 모델 재학습

lr2 = LogisticRegression(random_state = 0)
lr2.fit(x_train, y_train)
y_pred = lr2.predict(x_test)

# 분류 결과 평가

# +
# classification result for test set

print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('precision: %.2f' % precision_score(y_test, y_pred))
print('recall: %.2f' % recall_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))

# +
# confusion matrix

from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(confu, annot=True, annot_kws={'size':15}, cmap='OrRd', fmt='.10g')
plt.title('Confusion Matrix')
plt.show()
# -

# 이제 모델이 “긍정적인” 케이스와 “부정적인” 케이스를 모두 적당히 잘 맞춘 것을 확인할 수 있다.

# # 5. 긍정 / 부정 키워드 분석

# Logistic Regression 모델을 이용하여 긍/부정 키워드를 추출한다.
#
# 추출된 키워드를 통해서 이용자가 느끼는 호텔의 장,단점을 파악할 수 있고, 이를 기반으로 앞으로 유지해야 할 좋은 서비스와 개선이 필요한 아쉬운 서비스에 대해서도 어느정도 판단할 수 있다.

# Logistic Regression 모델에 각 단어의 coeficient를 시각화

lr2.coef_

# +
# print logistic regression's coef

plt.figure(figsize=(10, 8))
plt.bar(range(len(lr2.coef_[0])), lr2.coef_[0])
# -

# * 계수가 양인 경우 : 단어가 긍정적인 영향을 미쳤다
# * 계수가 음인 경우 : 단어가 부정적인 영향을 미쳤다
#     
# -> 이 계수들을 크기순으로 정렬하면, 긍정 / 부정 키워드를 출력하는 지표가 된다.

#  "긍정 키워드"와 "부정 키워드"의 Top 5를 각각 출력

print(sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse = True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse = True)[-5:])
# enumerate: 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환함 : 단어의 coeficient와 index가 출력

# 전체 단어가 포함한 "긍정 키워드 리스트"와 "부정 키워드 리스트"를 정의하고 출력

coef_pos_index = sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse = True)
coef_neg_index = sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse = False)
coef_pos_index

# index를 단어로 변환하여 "긍정 키워드 리스트"와 "부정 키워드 리스트"의 Top 20 단어를 출력

# +
invert_index_vectorizer = {v: k for k, v in cv.vocabulary_.items()}
invert_index_vectorizer

# invert_index_vectorizer = {v: k for k, v in word_token.to}
# invert_index_vectorizer
# -

len(invert_index_vectorizer)

for coef in coef_pos_index[:20]:
    print(invert_index_vectorizer[coef[1]], coef[0])

# **결론 : 이용객들이 주로 호텔의 청결도(clean)와 안락함(comfortable)에 만족하였다.**

for coef in coef_neg_index[:20]:
    print(invert_index_vectorizer[coef[1]], coef[0])


# **결론 : 이용객들이 주로 호텔이 청결하지 않으며 오래됐고 서비스가 무례하다며 불만족하였다.**

# # 6. 호텔리뷰 감성 (긍정/부정) 예측함수

# ## 6-1. Logistic Regression

def sentiment_predict_lr(input_text):

    #입력 텍스트에 대한 전처리 수행
    input_text = data_text_cleaning(input_text)
    #input_text = ["".join(input_text)]

    # 입력 텍스트의 피처 벡터화
    st_tfidf = cv.transform([input_text])
    tf_idf_vect = tfidf_vectorizer.transform(st_tfidf)
    tf_idf_vect = tf_idf_vect.toarray()

    # 최적 감성 분석 모델에 적용하여 감성 분석 평가
    score = float(lr2.predict(tf_idf_vect))

    #예측 결과 출력
    if(score > 0.5):
        print('예측 결과: ->> 긍정 감성')
    else :
        print('예측 결과: ->> 부정 감성')


sentiment_predict_lr('it was nice')

sentiment_predict_lr('it is not bad price')

sentiment_predict_lr('This was the second time we stayed at Bellagio in 6 months. We love the location. Easy access on foot to go shopping, dining, or shows. Room is very clean and great space. Many restaurants and high end shopping to enjoy. Several nice restaurants in the hotel. Yellowtail Restaurant... yummy!!!!')

#3점 리뷰
sentiment_predict_lr('I stayed in the Bellagio suite and still the house keeping service as well as the service desk fulfillment of requests was very bad. If you miss the regular room cleaning schedule…your room will not be cleaned even if you request multiple times')

#3점 리뷰
sentiment_predict_lr('Hotel over all is very nice. Aracelis went above and beyond to assist us...... Im a Amex Platinum card holder, an MGM rewards member, had reservations at Prime and Lago were we spend well over 1000 dollars. At the time of arrival, the person who helped us (Rosemary from Philippines) was not to helpful. She offered an upgarde to another room for an additional 450$ per night, even though as an Amex Platinum member we are supposed to receive this at no additional cost when available. On the second day of our stay, Aracelis was able to provide the service that was rightfully expected. In the original room that we were assigned (22 604) there was mold in the shower, dust on the furniture, drawers were dirty, shower head sprayed everywhere, there was glue and stains in the wall paper behind the bed specially. The quality of the toiletries was terrible')

# * comment : 3점리뷰에 대한 정확한 감성(긍정/부정) 예측을 위해서 더 많은 데이터가 필요함 

# AUC Score 확인
from sklearn.metrics import roc_auc_score

# +
# classification result for test set
print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('precision: %.2f' % precision_score(y_test, y_pred))
print('recall: %.2f' % recall_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))

AUC_lr = roc_auc_score(y_test,y_pred)
print('AUC: %.8f' % AUC_lr)
# -

# ### 그리드 서치
#
# 하이퍼파라미터를 튜닝하여 일반화 성능을 개선 
#
# -> 그리드 서치로서 관심 있는 하이퍼파라미터들을 대상으로 가능한 모든 조합을 시도

# 데이터를 train, validation, test set으로 나누는 방법은 성능이 좋고 널리 사용된다.
# 그러나 데이터를 나누는 방법에 매우 민감하므로 일반화 성능을 더 잘 평가하기 위해서는 훈련세트와 검정세트를 한번만 나누지 않고 교차 검증(cross validation)을 사용해서 각 매개변수의 조합의 성능을 평가할 수 있다.
#
# * 하이퍼파라미터 탐색을 자동화해 주는 도구
# * 탐색할 매개변수를 나열하면 교차 검증을 수행하여 가장 좋은 검증 점수의 매개변수 조합을 선택, 마지막으로 이 매개변수 조합으로 최종 모델을 훈련

from sklearn.model_selection import GridSearchCV # 하이퍼 파라미터 최적화

clf = LogisticRegression(random_state=0)
params = {'C': [15, 18, 19, 20, 22]}
grid_cv = GridSearchCV(clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(x_train, y_train)

# 최적의 평가 파라미터는 grid_cv.best_estimator_에 저장됨
print(grid_cv.best_params_, grid_cv.best_score_)# 가장 적합한 파라메터, 최고 정확도 확인

# ## 6-2. SVM

# +
from sklearn.svm import SVC

svm=SVC()
svm.fit(x_train,y_train)


# -

def sentiment_predict_svm(input_text):

    #입력 텍스트에 대한 전처리 수행
    input_text = data_text_cleaning(input_text)
    #input_text = ["".join(input_text)]

    # 입력 텍스트의 피처 벡터화
    st_tfidf = cv.transform([input_text])
    tf_idf_vect = tfidf_vectorizer.transform(st_tfidf)

    # 최적 감성 분석 모델에 적용하여 감성 분석 평가
    st_predict = float(svm.predict(st_tfidf))

    #예측 결과 출력
    if(st_predict == 0):
        print('예측 결과: ->> 부정 감성')
    else :
        print('예측 결과: ->> 긍정 감성')


sentiment_predict_svm('it was nice')

sentiment_predict_svm('Hotel over all is very nice. Aracelis went above and beyond to assist us...... Im a Amex Platinum card holder, an MGM rewards member, had reservations at Prime and Lago were we spend well over 1000 dollars. At the time of arrival, the person who helped us (Rosemary from Philippines) was not to helpful. She offered an upgarde to another room for an additional 450$ per night, even though as an Amex Platinum member we are supposed to receive this at no additional cost when available. On the second day of our stay, Aracelis was able to provide the service that was rightfully expected. In the original room that we were assigned (22 604) there was mold in the shower, dust on the furniture, drawers were dirty, shower head sprayed everywhere, there was glue and stains in the wall paper behind the bed specially. The quality of the toiletries was terrible')

sentiment_predict_svm('Subpar Dirty hallways and less than average room.')

# predict in test set
y_pred_svm = svm.predict(x_test)

# +
# classification result for test set
print('accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('precision: %.2f' % precision_score(y_test, y_pred_svm))
print('recall: %.2f' % recall_score(y_test, y_pred_svm))
print('F1: %.2f' % f1_score(y_test, y_pred_svm))

AUC_svm = roc_auc_score(y_test,y_pred_svm)
print('AUC: %.8f' % AUC_svm)
# -

# ### 그리드 서치

clf = SVC(random_state=0)
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}
grid_cv = GridSearchCV(clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv.fit(x_train, y_train)

# 최적의 평가 파라미터는 grid_cv.best_estimator_에 저장됨
print(grid_cv.best_params_, grid_cv.best_score_)# 가장 적합한 파라메터, 최고 정확도 확인

# ## 6-3. Decision Tree

# +
from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)


# -

def sentiment_predict_dt(input_text):

    #입력 텍스트에 대한 전처리 수행
    input_text = data_text_cleaning(input_text)
    #input_text = ["".join(input_text)]

    # 입력 텍스트의 피처 벡터화
    st_tfidf = cv.transform([input_text])
    tf_idf_vect = tfidf_vectorizer.transform(st_tfidf)

    # 최적 감성 분석 모델에 적용하여 감성 분석 평가
    st_predict = float(dt.predict(st_tfidf))

    #예측 결과 출력
    if(st_predict > 0.5):
        print('예측 결과: ->> 부정 감성')
    else :
        print('예측 결과: ->> 긍정 감성')


sentiment_predict_dt('it was nice')

sentiment_predict_dt('Hotel over all is very nice. Aracelis went above and beyond to assist us...... Im a Amex Platinum card holder, an MGM rewards member, had reservations at Prime and Lago were we spend well over 1000 dollars. At the time of arrival, the person who helped us (Rosemary from Philippines) was not to helpful. She offered an upgarde to another room for an additional 450$ per night, even though as an Amex Platinum member we are supposed to receive this at no additional cost when available. On the second day of our stay, Aracelis was able to provide the service that was rightfully expected. In the original room that we were assigned (22 604) there was mold in the shower, dust on the furniture, drawers were dirty, shower head sprayed everywhere, there was glue and stains in the wall paper behind the bed specially. The quality of the toiletries was terrible')

sentiment_predict_dt('Subpar Dirty hallways and less than average room.')

sentiment_predict_dt('Dirty Room Stayed at the Bellagio for the first time ever for a work trip.')

# predict in test set
y_pred_dt = dt.predict(x_test)

# +
# classification result for test set
print('accuracy: %.2f' % accuracy_score(y_test, y_pred_dt))
print('precision: %.2f' % precision_score(y_test, y_pred_dt))
print('recall: %.2f' % recall_score(y_test, y_pred_dt))
print('F1: %.2f' % f1_score(y_test, y_pred_dt))

AUC_dt = roc_auc_score(y_test,y_pred_dt)
print('AUC: %.8f' % AUC_dt)
# -

# ### 그리드 서치

clf = DecisionTreeRegressor(random_state=0)
# min_impurity_decrease : 최소 불순도(지니계수)가 낮다 = 엔트로피가 낮다 = 데이터의 균일도가 높다  
params = {"max_depth": [1, 5, 10],
          "min_samples_split": [2, 3],
          "min_impurity_decrease": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
grid_cv = GridSearchCV(clf, param_grid=params)
grid_cv.fit(x_train, y_train)

# 최적의 평가 파라미터는 grid_cv.best_estimator_에 저장됨
print(grid_cv.best_params_, grid_cv.best_score_)# 가장 적합한 파라메터, 최고 정확도 확인

# ## 6-4. Random Forest

# +
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# -

def sentiment_predict_rf(input_text):

    #입력 텍스트에 대한 전처리 수행
    input_text = data_text_cleaning(input_text)
    #input_text = ["".join(input_text)]

    # 입력 텍스트의 피처 벡터화
    st_tfidf = cv.transform([input_text])
    tf_idf_vect = tfidf_vectorizer.transform(st_tfidf)

    # 최적 감성 분석 모델에 적용하여 감성 분석 평가
    st_predict = float(rf.predict(st_tfidf))

    #예측 결과 출력
    if(st_predict > 0.5):
        print('예측 결과: ->> 부정 감성')
    else :
        print('예측 결과: ->> 긍정 감성')


sentiment_predict_rf('Hotel over all is very nice. Aracelis went above and beyond to assist us...... Im a Amex Platinum card holder, an MGM rewards member, had reservations at Prime and Lago were we spend well over 1000 dollars. At the time of arrival, the person who helped us (Rosemary from Philippines) was not to helpful. She offered an upgarde to another room for an additional 450$ per night, even though as an Amex Platinum member we are supposed to receive this at no additional cost when available. On the second day of our stay, Aracelis was able to provide the service that was rightfully expected. In the original room that we were assigned (22 604) there was mold in the shower, dust on the furniture, drawers were dirty, shower head sprayed everywhere, there was glue and stains in the wall paper behind the bed specially. The quality of the toiletries was terrible')

sentiment_predict_rf('Dirty hallways')

sentiment_predict_rf('Dirty Room Stayed at the Bellagio for the first time ever for a work trip.')

# predict in test set
y_pred_rf = rf.predict(x_test)

# +
# classification result for test set
print('accuracy: %.2f' % accuracy_score(y_test, y_pred_rf))
print('precision: %.2f' % precision_score(y_test, y_pred_rf))
print('recall: %.2f' % recall_score(y_test, y_pred_rf))
print('F1: %.2f' % f1_score(y_test, y_pred_rf))

AUC_rf = roc_auc_score(y_test,y_pred_rf)
print('AUC: %.8f' % AUC_rf)
# -

# ### 그리드 서치

clf = RandomForestClassifier(random_state=0)
# min_impurity_decrease : 최소 불순도(지니계수)가 낮다 = 엔트로피가 낮다 = 데이터의 균일도가 높다  
params = {"max_depth": [1, 5, 10],
          "min_samples_split": [2, 3],
          "min_impurity_decrease": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
grid_cv = GridSearchCV(clf, param_grid=params)
grid_cv.fit(x_train, y_train)

# 최적의 평가 파라미터는 grid_cv.best_estimator_에 저장됨
print(grid_cv.best_params_, grid_cv.best_score_)# 가장 적합한 파라메터, 최고 정확도 확인

# ## 6-5. KNN

# +
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor()
knn.fit(x_train,y_train)


# -

def sentiment_predict_knn(input_text):

    #입력 텍스트에 대한 전처리 수행
    input_text = data_text_cleaning(input_text)
    #input_text = ["".join(input_text)]

    # 입력 텍스트의 피처 벡터화
    st_tfidf = cv.transform([input_text])
    tf_idf_vect = tfidf_vectorizer.transform(st_tfidf)

    # 최적 감성 분석 모델에 적용하여 감성 분석 평가
    st_predict = float(knn.predict(st_tfidf))

    #예측 결과 출력
    if(st_predict == 0):
        print('예측 결과: ->> 부정 감성')
    else :
        print('예측 결과: ->> 긍정 감성')


sentiment_predict_knn('it was nice')

sentiment_predict_knn('Hotel over all is very nice. Aracelis went above and beyond to assist us...... Im a Amex Platinum card holder, an MGM rewards member, had reservations at Prime and Lago were we spend well over 1000 dollars. At the time of arrival, the person who helped us (Rosemary from Philippines) was not to helpful. She offered an upgarde to another room for an additional 450$ per night, even though as an Amex Platinum member we are supposed to receive this at no additional cost when available. On the second day of our stay, Aracelis was able to provide the service that was rightfully expected. In the original room that we were assigned (22 604) there was mold in the shower, dust on the furniture, drawers were dirty, shower head sprayed everywhere, there was glue and stains in the wall paper behind the bed specially. The quality of the toiletries was terrible')

sentiment_predict_knn('Subpar Dirty hallways and less than average room.')

sentiment_predict_knn('Dirty Room Stayed at the Bellagio for the first time ever for a work trip.')

# predict in test set
y_pred_knn = knn.predict(x_test)

# +
# classification result for test set
# print('accuracy: %.2f' % accuracy_score(y_test, y_pred_knn))
# print('precision: %.2f' % precision_score(y_test, y_pred_knn))
# print('recall: %.2f' % recall_score(y_test, y_pred_knn))
# print('F1: %.2f' % f1_score(y_test, y_pred_knn))

# AUC Score 확인
AUC_knn = roc_auc_score(y_test,y_pred_knn)
print('AUC: %.8f' % AUC_knn)
# -

# ### 그리드 서치

# https://blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=baek2sm&logNo=221763552440&redirect=Dlog&widgetTypeCall=true&directAccess=false

clf = KNeighborsRegressor()
# min_impurity_decrease : 최소 불순도(지니계수)가 낮다 = 엔트로피가 낮다 = 데이터의 균일도가 높다  
params = [{'n_neighbors' : range(3,20)}]
grid_cv = GridSearchCV(clf, param_grid=params)
grid_cv.fit(x_train, y_train)

# 최적의 평가 파라미터는 grid_cv.best_estimator_에 저장됨
print(grid_cv.best_params_, grid_cv.best_score_)# 가장 적합한 파라메터, 최고 정확도 확인

# ## 6-6. LSTM

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

vocab_size = len(word_token)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64)

# ## 6-7. 각 모델별 AUC score 비교

AUC_compare = [0.886423, 0.847511, 0.75893, 0.81440, 0.518946, 0.827861, 0.812024, 0.850780, 0.5]
AUC_compare

# +
plt.bar(range(len(AUC_compare)), AUC_compare)
plt.title("AUC score per model") #차트 제목
plt.ylabel('AUC')
plt.ylim([0, 1])
plt.xticks([0,1,2,3,4, 5, 6, 7, 8], ['LR', 'SVM', 'DT', 'RF', 'KNN', 'GBM', 'XGB', 'CB', 'LSTM'])
plt.figure(figsize=(300,30))

plt.show()
# -

# # 7. 개선점

# 1. 긍정/부정 키워드에 대한 워드 클라우드를 생성하여 결과를 보다 가시적으로 표현할 것
# 2. RNN 기반의 LSTM 모델을 추가하여 정확도를 보다 높일 것, 이때 평가 지표로 AUC score 대신 다른 것을 생각해 볼 것 
