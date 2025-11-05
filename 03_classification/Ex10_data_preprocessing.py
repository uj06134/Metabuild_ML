import numpy as np
import pandas as pd

data = {
    "Name" : ['kim', 'park', 'choi', 'choi', 'kim', np.nan, 'park'],
    "Age" : [30, 30, 25, np.nan, 30, 20, 29],
    "Addr": ['Jeju', np.nan, 'Jeju', 'Seoul', 'Jeju', 'Seoul', 'Busan']
}
df = pd.DataFrame(data)
print(df)

# 중복 삭제
df_duplicate = df.drop_duplicates()
print(df_duplicate)
print()

# 중복 컬럼 삭제
df_duplicate_name = df.drop_duplicates(subset=['Name'])
print(df_duplicate_name)
print()

# 중복 컬럼 삭제(마지막 유지)
df_duplicate_name_last = df.drop_duplicates(subset=['Name'], keep="last")
print(df_duplicate_name_last)
print()

# 결측치 제거
df_na = df.dropna()
print(df_na)
print()

# 결측치 제거(이름, 나이)
df_na_name_age = df.dropna(subset=['Name', 'Age'])
print(df_na_name_age)
print()

# 중복 개수
dup_count = sum(df.duplicated())
print("중복 개수:", dup_count)

dupe_mask = df.duplicated()
print(dupe_mask)
print()