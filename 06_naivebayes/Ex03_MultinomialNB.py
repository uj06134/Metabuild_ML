from konlpy.tag import Okt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sample = "오늘 일정 확인"
okt = Okt()
result = okt.morphs(sample)
# print("result:", result)

result2 = okt.morphs("한국어는 주변 언어와 어떤 친족 관계도 밝혀지지 않은 언어다.")
# print("result2:", result2)

print("--------------------------------")
df = pd.read_csv("../00_dataIn/mailList.csv", encoding="utf-8")
# print(df.head())

emails = [tuple(row) for row in df.itertuples(index=False)]
# print('emails:\n', emails)

def tokenize(text):
    return ' '.join(okt.morphs(text))

emails_tokenized = [(tokenize(subject),label) for subject, label in emails]
# print('emails_tokenized:\n', emails_tokenized)

# *: 리스트 풀기(unpacking)
x, y = zip(*emails_tokenized)
# print('x:\n', x)
# print('y:\n', y)

x_train_text, x_test_text, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 학습용 데이터로 단어 사전 생성
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train_text)
print("vocabulary:",vectorizer.vocabulary_)

# 테스트 데이터
x_test = vectorizer.transform(x_test_text)
print(x_test_text[0])
print(x_test)
print(y_test)
print(vectorizer.get_feature_names_out())

# 나이브 베이즈: 확률에 기반한 분류 알고리즘
# MultinomialNB: 단어 등장 횟수를 기반
model = MultinomialNB()
model.fit(x_train, y_train)
predict_proba = model.predict_proba(x_test)
print('predict_proba:\n', predict_proba)

# 예측
pred = model.predict(x_test)
print('pred:\n', pred)

print('----------------------------')
fp = open("../00_dataIn/checkedMail.csv", encoding="utf-8")
new_data = [onemail.strip() for onemail in fp.readlines()]
print(new_data)

fp.close()

final_email_info = []
for new_email in new_data:
    new_email_tokenized = tokenize(new_email)
    new_email_vec = vectorizer.transform([new_email_tokenized])
    # print(new_email_vec)
    pred2 = model.predict(new_email_vec)
    result = f"{new_email} : {pred2[0]}"
    final_email_info.append(result)

print(final_email_info)