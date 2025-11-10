from collections import Counter
import numpy as np

ball_list = ['red', 'blue', 'red', 'blue', 'red', 'blue', 'red']
counter = Counter(ball_list)
print('counter:', counter)
print('counter.values():', counter.values())
print('sum(counter.values()):', sum(counter.values()))
print('counter.items():', counter.items())
total_cnt = sum(counter.values())

# 색상별 비율 계산 : 개수 ÷ 전체 개수
ratios = {key : val / total_cnt for key, val in counter.items()}
print('ratios:', ratios)

probabilities = [value for value in ratios.values()]
print('probabilities:', probabilities)

# 엔트로피(Entropy)를 계산하는 함수 정의
# 엔트로피 공식: H = -Σ(p * log2(p))
def entropy_test(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 두 색의 불확실성(무질서도) 정도
# 확률이 균등 -> 엔트로피(불확실성)가 높다
entropy = entropy_test(probabilities)
print('entropy:', entropy)

