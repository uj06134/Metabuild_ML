from sklearn.feature_extraction.text import CountVectorizer

words = []
with open('../00_data_in/text01.txt', 'r' ,encoding='utf-8') as f:
    for line in f:
        words.append(line.strip())

print(words)

vectorizer = CountVectorizer(min_df=2, stop_words=['세일'])
x = vectorizer.fit_transform(words)
print("vocabulary:", vectorizer.vocabulary_)
print(x)
print(x.toarray())