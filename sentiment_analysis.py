import string
from warnings import filterwarnings
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from PIL.ImagePalette import random
from nltk.corpus import stopwords, words
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

"""
Task 1: Metin Ön İşleme
    Step 1: amazon.xlsx verisini okutunuz
    Step 2:Review değişkeni üzerinde ;
        -Tüm harfleri küçük harfe çeviriniz.
        -Noktalama işaretlerini çıkarınız.
        -Yorumlarda bulunan sayısal ifadeleri çıkarınız.
        -Bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız.
        -1000'den az geçen kelimeleri veriden çıkarınız.
        -Lemmatization işlemini uygulayınız
"""

df = pd.read_excel("C:/Users/SALİH KARAYILAN/Desktop/AmazonDataset/amazon.xlsx")
print('\nHAM DATASET :\n\n', df.head())

# Normalizing case folding (Tüm harfleri küçük harfe çeviriniz)

df['Review'] = df['Review'].str.lower()
print('\n NORMALIZING CASE FOLDING :\n\n', df['Review'])

# Puntuations (Noktalama işaretlerini çıkarınız.)

df['Review'] = df['Review'].str.replace(r'[^\w\s]', '', regex=True)
print('\n PUNCTUATIONS :\n\n', df['Review'])

# numbers (Yorumlarda bulunan sayısal ifadeleri çıkarınız.)

df['Review'] = df['Review'].str.replace(r'\d', '', regex=True)
print('\n NUMBERS :\n\n', df['Review'])

# stopwords (Bilgi içermeyen kelimeleri veriden çıkarınız)
nltk.download('stopwords')

sw = stopwords.words('english')
print('\n STOPWORDS :\n\n', sw)

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
print('\n REMOVE STOPWORDS :\n\n', df['Review'])

# Rare words (1000'den az geçen kelimeleri veriden çıkarınız.)

temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()
print('\n WORDS AND THEM NUMBERS:\n\n', temp_df)
drops = temp_df[temp_df < 100]
print('\n WORD  NUMBERS <100 :\n\n', drops)

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
print('\n WITHOUT RARE WORDS :\n\n', df['Review'])

# Lemmatization (Kelimeleri köklerine ayırma işlemidir. Takıları kaldırma) (stemming de vardır )

nltk.download('wordnet')

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print('\n LEMMATIZATION :\n\n', df['Review'])

"""
Task 2: Text Visualization (Metin Görselleştirme)

    Step 1: Barplot görselleştirme işlemi için;
        - "Review" değişkeninin içerdiği kelimelerin frekanslarını hesaplayınız, tf olarak kaydediniz.
        - tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
        - "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini
           mamlayınız.
    Step 2: WordCloud görselleştirme işlemi için;
        - "Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz.
        - WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz.
        - Kaydettiğiniz wordcloud'u ilk adımda oluşturduğunuz string ile generate ediniz.
        - Görselleştirme adımlarını tamamlayınız. (figure, imshow, axis, show)
"""

# Terim Frekanslarının Hesaplanması ("Review" değişkeninin içerdiği kelimelerin frekanslarını hesaplayınız, tf olarak kaydediniz.)

tf = df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

# tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
tf.columns = ["words", "tf"]
print('\n ASCENDING OF FREQUENCY WORDS :\n\n', tf.sort_values("tf", ascending=True))

# Barplot (tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini tamamlayınız.)

tf[tf['tf'] > 500].plot.bar(x="words", y='tf')
plt.show()

# WordCloud ("Review" değişkeninin içerdiği tüm kelimeleri "text" isminde string olarak kaydediniz.)

text = " ".join(i for i in df.Review)
print('\n TEXT :\n\n' ,text)

# WordCloud kullanarak şablon şeklinizi belirleyip kaydediniz.

wordcloud = WordCloud(max_font_size=500,
                      max_words=100,
                      background_color="white",
                      ).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.show()
wordcloud.to_file("wordcloud.png")

# Şablonlara Göre Wordcloud

sql_mask = np.array(Image.open("Sql_data_base_with_logo.png"))

wc = WordCloud(background_color='orange',
               max_words=1000,
               mask=sql_mask,
               contour_width=5,
               contour_color='black').generate(text)
plt.figure(figsize=[10,10])
plt.imshow(wc ,interpolation= 'bilinear')
plt.axis('off')
plt.show()
wordcloud.to_file('template_sql.png')

"""
Task 3 : Sentiment Analysis (Duygu Analizi):
    Step 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz.
    Step 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarını inceleyiniz;
        -"Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız.
        -İncelenen ilk 10 gözlem için compund skorlarına göre filtreleyerek tekrar gözlemleyiniz.
        -10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz.
        -"Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e
          ekleyiniz.
    NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken
         oluşturulmuş oldu.
"""

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Example for SentimentIntensityAnalyzer object

x = sia.polarity_scores('I liked this music but it isnot good as the other one.')
print('\n polarity scores of x statement :\n\n', x)

# {'neg': 0.0, 'neu': 0.61, 'pos': 0.39, 'compound': 0.6956}

# Feature Engineering

a = df['Review'][0:10].apply(lambda x: sia.polarity_scores(x))
print(df['Review'][0:10])
print('\n polarity scores of a statement :\n\n', a)

b = df['Review'][0:10].apply(lambda x: sia.polarity_scores(x)['compound'])
print('\n compound score of b statement :\n\n', b)

df['polarity_score'] = df['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
print('\n polarity scores all of review column in dataset :\n\n', df['polarity_score'])

x = df['Review'][0:10].apply(lambda x: 'pos' if sia.polarity_scores(x)['compound'] > 0 else 'neg')
print('\n labeling first 10 words neg-pos using polarity scores \n\n:', x)

df['sentiment_label'] = df['Review'].apply(lambda x: 'pos' if sia.polarity_scores(x)['compound'] > 0 else 'neg')
print('\n labeling all reviews neg-pos using polarity scores \n\n:', df['sentiment_label'])

"""
Task 4: Preparation of ML
    Step 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olarak ayırınız.
    Step 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;
        - TfidfVectorizer kullanarak bir nesne oluşturunuz.
        - Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
        - Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.

"""

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["sentiment_label"],
                                                    random_state=42)

# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Examples

'''
CountVectorizer:
Transforms text into a sparse matrix of n-gram counts.

TfidfTransformer:
Performs the TF-IDF transformation from a provided matrix of counts.
'''

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# word frequency

vectorizer = CountVectorizer()
A_c = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
# >> ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
print(A_c.toarray())

'''
 [0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]
'''

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
A2 = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
print(vectorizer.get_feature_names_out())
'''
array(['and this', 'document is', 'first document', 'is the', 'is this',
       'second document', 'the first', 'the second', 'the third', 'third one',
       'this document', 'this is', 'this the'], ...)
'''

print(A2.toarray())

'''
 [0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]
'''
# ngram frequency

vectorizer = TfidfVectorizer()
A = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
print(vectorizer.get_feature_names_out())
print(A.toarray())

'''
 [0.         0.46979139     0.58028582      0.38408524 0.         0.             0.38408524     0.         0.38408524]
 [0.         0.6876236      0.              0.28108867 0.         0.53864762     0.28108867     0.         0.28108867]
 [0.51184851 0.             0.              0.26710379 0.51184851 0.             0.26710379     0.51184851 0.26710379]
 [0.         0.46979139     0.58028582      0.38408524 0.         0.             0.38408524     0.         0.38408524]


'''
# I will continue use Amazon dataset


tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

'''
Task 5: Modelleme (Lojistik Regresyon):

    Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.

    Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz;
        -Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
        -classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
        -cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.

    Adım 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması;
        -sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atayınız.
        -Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
        -Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
        -Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
        -Örneklemi ve tahmin sonucunu ekrana yazdırınız.

'''
#  Logistic Regression


from sklearn.metrics import classification_report

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred, test_y))

a = cross_val_score(log_model,
                    x_test_tf_idf_word,
                    test_y,
                    cv=5).mean()
print(a)

# 0.925866802236909 ORANLA DOĞRU TAHMİN YAPILACAKTIR.


first_review = pd.Series('that is bad')
second_review = pd.Series('you are very stubborn')

first_review = TfidfVectorizer().fit(train_x).transform(first_review)
print(log_model.predict(first_review))

second_review = TfidfVectorizer().fit(train_x).transform(second_review)
print(log_model.predict(second_review))

random_review = pd.Series(df['Review'].sample(1).values)
print(random_review)
random_review = TfidfVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(random_review)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')

'''

Task 6 : Random Forest
    Step 1: Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
        - RandomForestClassifier modelini kurup fit ediniz.
        - Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
        - Lojistik regresyon modeli ile sonuçları karşılaştırınız.
'''

# Featute Üretme

# TF IDF Word-Level
rf_model2 = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
c = cross_val_score(rf_model2, x_test_tf_idf_word, test_y, cv=5).mean()
print("TF IDF Word-Level : ", c)

# TF IDF ngram

rf_model3 = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
d = cross_val_score(rf_model3, x_test_tf_idf_word, test_y, cv=5).mean()
print("TF IDF ngram : ", d)

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
a = cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5).mean()
print(a)

# 0.9672191154041687 oranla doğru tahmin


