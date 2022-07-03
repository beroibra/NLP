from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from emoticons import EMOTICONS
from collections import Counter
import re, string
import emoji
import spacy



def tokenize(df, nlp):
    df['preprocessing'] = df['sms'].apply(lambda x: nlp.tokenizer(x).text)


def lower_text(df):
    df['preprocessing'] = df["preprocessing"].str.lower()


def number_removal(df):
    df['preprocessing'] = df["preprocessing"].str.replace('\d+', '')


def rm_punctuation(txt):
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_punctuation(df):
    df['preprocessing'] = df['preprocessing'].apply(lambda x: rm_punctuation(x))


def remove_emojis(df):
    df['preprocessing'] = df['preprocessing'].apply(lambda x: emoji.demojize(x))


def rm_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)


def remove_emoticons(df):
    df['preprocessing'] = df['preprocessing'].apply(lambda x: rm_emoticons(x))


def strip_text(df):
    df['preprocessing'] = df["preprocessing"].str.strip()


def rm_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_urls(df):
    df['preprocessing'] = df['preprocessing'].apply(lambda x: rm_urls(x))


def rm_stopwords(txt):
    return " ".join([word for word in str(txt).split() if word not in set(stopwords.words('english'))])


def remove_stopwords(df):
    df['swr'] = df['preprocessing'].apply(lambda x: rm_stopwords(x))


def freq_word_counter(df):
    cnt = Counter()
    for text in df["swr"].values:
        for word in text.split():
            cnt[word] += 1

    return cnt


def rm_freqwords(text, cnt, word_num):
    return " ".join([word for word in str(text).split() if word not in
                     set([w for (w, wc) in cnt.most_common(word_num)])])


def remove_freqwords(df, cnt, word_num):
    df["freq_rare_word_rm"] = df["swr"].apply(lambda x: rm_freqwords(x, cnt, word_num))


def rm_rarewords(text, cnt, num_word):
    return " ".join([word for word in str(text).split() if word not in
                     set([w for (w, wc) in cnt.most_common()[:-num_word - 1:-1]])])


def remove_rarewords(df, cnt, num_word):
    df["freq_rare_word_rm"] = df["swr"].apply(lambda x: rm_rarewords(x, cnt, num_word))


def stem_words(text, stemmer_func):
    return " ".join([stemmer_func.stem(word) for word in text.split()])


def stemmer(df, stemmer_func):
    df["stemmed"] = df["freq_rare_word_rm"].apply(lambda text: stem_words(text, stemmer_func))


def lemmatize_words(text, lemmatizer_func):
    return " ".join([lemmatizer_func.lemmatize(word) for word in text.split()])


def lemmatizer(df, lemmatizer_func):
    df["lemmatized"] = df["freq_rare_word_rm"].apply(lambda text: lemmatize_words(text, lemmatizer_func))



def generate_processed_df(df):
    # 1) basic pre-processing df["preprocessing"]
    # Tokenize
    # Lower Casing
    # Number Removal
    # Strip Text
    # Extra White space removel
    # Emoji Removal
    # Emoticons removal
    # Punctuation Removal
    # URL removal
    nlp = spacy.load('en_core_web_trf')

    tokenize(df, nlp)
    lower_text(df)
    strip_text(df)
    number_removal(df)
    remove_punctuation(df)
    remove_urls(df)
    remove_emoticons(df)

    # 2) basic pre-processing + Stop word removal df['swr']
    remove_stopwords(df)

    # 3) basic pre-processing + Stop word removal + Frequent/rare word removal df["freq_rare_word_rm"]
    cnt = freq_word_counter(df)
    remove_freqwords(df, cnt, 7)
    remove_rarewords(df, cnt, 7)

    # 4) basic pre-processing + Stop word removal + Frequent/rare word removal + Stemming df["stemmed"]
    stemmer_func = PorterStemmer()

    stemmer(df, stemmer_func)

    # 5) basic pre-processing + Stop word removal + Frequent/rare word removal + lemmatization df["lemmatized"]
    lemmatizer_func = WordNetLemmatizer()
    lemmatizer(df, lemmatizer_func)