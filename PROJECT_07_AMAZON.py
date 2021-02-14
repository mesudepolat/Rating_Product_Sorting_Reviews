###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################


import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


###################################################
# GÖREV 1: Bir ürünün rating'ini güncel yorumlara göre hesaplayınız ve eski rating ile kıyaslayınız.
###################################################

df_ = pd.read_csv("datasets/df_sub.csv")
df = df_.copy()
df.head()
df.info()
df.shape

# ürünün ortalama puanı
df["overall"].mean()
# 4.587589013224822

# tarihe göre ağırlıklı puan ortalaması
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
df['reviewTime'].max()
current_date = pd.to_datetime('2014-12-08 0:0:0')

# yorum sonrası ne kadar gün geçmiş
df["day_diff"] = (current_date - df['reviewTime']).dt.days
df["day_diff"].head()

# Zamanın çeyrek değerlere göre bölünmesi
a = df["day_diff"].quantile(0.25)
b = df["day_diff"].quantile(0.50)
c = df["day_diff"].quantile(0.75)

# a,b,c değerlerine göre ağırlıklı puan hesaplama

df.loc[df["day_diff"] <= a, "overall"].mean() * 28 / 100 + \
    df.loc[(df["day_diff"] > a) & (df["day_diff"] <= b), "overall"].mean() * 26 / 100 + \
    df.loc[(df["day_diff"] > b) & (df["day_diff"] <= c), "overall"].mean() * 24 / 100 + \
    df.loc[(df["day_diff"] > c), "overall"].mean() * 22 / 100
# 4.595593165128118

###################################################
# Görev 2: Görüntülenecek ilk 20 yorumu belirleyiniz
###################################################

# helpful değişkeni içerisinden 3 değişken türetme
# 1: helpful_yes, 2: helpful_no,  3: total_vote
df['helpful_yes'] = df[['helpful']].applymap(lambda x : x.split(', ')[0].strip('[')).astype(int)
df['total_vote'] = df[['helpful']].applymap(lambda x : x.split(', ')[1].strip(']')).astype(int)
df['helpful_no'] = df['total_vote'] - df['helpful_yes']
df.iloc[25:30]

# score_pos_neg_diff'a göre skorlar oluşturma
def score_pos_neg_diff(pos, neg):
    return pos - neg

df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                      x["helpful_no"]),
         axis=1)


# score_pos_neg_diff ismiyle veri setine kaydetme
df['score_pos_neg_diff'] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]),
                                    axis=1)


# score_average_rating'a göre skorlar oluşturma
def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)


# score_average_rating ismiyle veri setine kaydetme
df['score_average_rating'] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]),
                                      axis=1)

# wilson_lower_bound'a göre skorlar oluşturma ve veri setine wilson_lower_bound ismiyle kaydetme.
def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]),
                                    axis=1)


# Ürün sayfasında gösterilecek 20 yorumu belirleyiniz ve sonuçları yorumlayınız.
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Yorumları wilson_lower_bound değişkenine göre sıralamak müşteriler için en faydalı sonucu verecektir.
# Sıralama yaptıktan sonra ürün rating'i 1 olan yourmlar da gelmiştir böylece yorumları okuyan bir müşteri
# sıralamanın objektif olduğuna inanır ve tercihini güvenerek yapar.
