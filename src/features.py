from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def bow_features(train, test):
    vector = CountVectorizer()
    return vector.fit_transform(train), vector.transform(test), vector

def tf_idf_features(train, test):
    vector = TfidfVectorizer()
    return vector.fit_transform(train), vector.transform(test), vector

def n_gram_features(train, test):
    vector = TfidfVectorizer(ngram_range=(1, 4))
    return vector.fit_transform(train), vector.transform(test), vector
