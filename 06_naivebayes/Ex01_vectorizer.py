from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "banana apple apple q ??",
    "banana grape w q@",
    "grape apple banana 123",
]

# 문장 -> 벡터
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(docs)
# vectorizer.fit(docs)
# x = vectorizer.transform(docs)
print("vocabulary:", vectorizer.vocabulary_)
print(x.shape)
print(x)

feature_names = vectorizer.get_feature_names_out()
print("feature_names:\n", feature_names)
print("type(feature_names):\n", type(feature_names))
index = list(feature_names).index("apple")
print("index:", index)
print(x.toarray())
print("-----------------------------------")
docs = [
    'green red blue',
    'blue red yellow red red' ,
    'red red blue blue'
]

vectorizer.fit(docs)

x=vectorizer.transform(docs)
print(x.toarray())