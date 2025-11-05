import pandas as pd

data = {
    'Color':['Red', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'M', 'S'],
    'Price': [10, 15, 20, 15, 10],
    'Category' : ['Shoes','Shirts','Pants','Shoes','Shirts']
}

df = pd.DataFrame(data)

# 모든 컬럼 표시
pd.set_option('display.max_columns', None)

# 원-핫 인코딩(One-hot Encoding): 범주형 데이터를 기계가 인식할 수 있는 숫자 형태로 변환하는 방식
# 문자열 컬럼만 자동 인코딩
df_encode = pd.get_dummies(df)
print(df_encode)
