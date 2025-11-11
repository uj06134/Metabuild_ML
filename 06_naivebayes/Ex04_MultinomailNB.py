from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

titles = [
    # 정치
    "대통령 신년 기자회견 열려", "야당 대표, 정부 정책 비판", "정치권, 총선 준비 돌입",
    "국회, 예산안 처리 본격화", "총리, 경제 정책 발표", "정당 간 협상 난항 지속",
    "대선 후보, 공약 발표", "국회의원 선거법 개정 논의", "정부, 대북 정책 강화",

    # 스포츠
    "한국, 월드컵 본선 진출 확정", "류현진, 시즌 첫 승 기록", "손흥민 멀티골로 팀 승리 견인",
    "김민재, 수비수 최초 유럽 리그 우승", "박지성, 전설적인 축구 선수 은퇴", "프로야구, 새 시즌 개막",
    "축구 대표팀, 친선 경기 승리", "테니스 선수, 그랜드슬램 우승", "골프 대회, 상금 기록 경신",

    # 경제
    "주식 시장, 3일 연속 하락", "환율 급등에 수출 기업 비상", "부동산 가격 하락세 지속",
    "코스피, 2500선 회복", "금리 인상에 대출 부담 증가", "수출 호조로 무역 흑자 확대",
    "소비자 물가 상승률 2% 돌파", "은행, 신규 대출 규제 강화", "중소기업 지원 정책 발표"
]

labels = [
    # 정치
    "politics", "politics", "politics",
    "politics", "politics", "politics",
    "politics", "politics", "politics",

    # 스포츠
    "sports", "sports", "sports",
    "sports", "sports", "sports",
    "sports", "sports", "sports",

    # 경제
    "economy", "economy", "economy",
    "economy", "economy", "economy",
    "economy", "economy", "economy"
]
# 형태소 분석
okt = Okt()

# 필요한 품사들만 토큰화
def tokenize(text):
    tokens = okt.pos(text)
    return ' '.join([word for word, pos in tokens if pos in ['Noun', 'Verb', 'Adjective']])

titles_tokenized = [tokenize(t) for t in titles]
# print(tokenized_titles)

# 벡터화
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(titles_tokenized)
y = labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 모델 학습
model = MultinomialNB()
model.fit(x_train, y_train)
predict_proba = model.predict_proba(x_test)
print('predict_proba:\n', predict_proba)

# 예측
pred = model.predict(x_test)
print('pred:\n', pred)
print('y_test:\n', y_test)

# 혼동행렬
cm = confusion_matrix(y_test, pred)
print('cm:\n', cm)

score = model.score(x_test, y_test)
print('score:', score)
acc = accuracy_score(y_test, pred)
print('acc:', acc)

print('-------------------------')
# 새 데이터
new_titles = [
     "대통령, 외교 정상회담 진행",        # politics
    "코스피, 2600선 돌파하며 상승세",     # economy
    "김민재, 유럽 축구 리그 우승 기록",    # sports
    "환율 변동성 커져 수출업계 긴장",      # economy
    "야당, 새로운 총선 전략 발표",        # politics
    "손흥민, 시즌 20호 골 달성",          # sports
]

new_titles_tokenized = [tokenize(t) for t in new_titles]
new_x = vectorizer.transform(new_titles_tokenized)
new_pred = model.predict(new_x)

for i, p in enumerate(new_pred):
    print(f"'{new_titles_tokenized[i]}' -> {p}")