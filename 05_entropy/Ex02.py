from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

df = sns.load_dataset('titanic')
df.rename(columns={'sex':'gender'}, inplace=True)
print(df.head())

df = df[['gender', 'class', 'survived']].dropna()
print(df.isnull().sum())

survived_counts = df['survived'].value_counts()
print('survived_counts:\n', survived_counts)

def entropy_test(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
result = []
for (gen, cls), group in df.groupby(['gender', 'class']):
    print('gender:', gen, 'class:', cls)
    print('group:', group)
    counts = Counter(group['survived'])
    print('counts:\n', counts)

    total_count = sum(counts.values())
    ratios = {key: value / total_count for key, value in counts.items()}
    print('ratios:', ratios)

    probabilities = [value for value in ratios.values()]
    print('probabilities:', probabilities)

    entropy = entropy_test(probabilities)
    print('entropy:', entropy)

    result.append({'성별':gen, '객실등급':cls, '엔트로피':entropy})
print(result)

entropy_df = pd.DataFrame(result)
print(entropy_df)

plt.figure(figsize=[8,5])
sns.barplot(data=entropy_df, x='객실등급', y='엔트로피', hue='성별')
plt.title('Titanic 생존여부의 엔트로피(성별/객실등급)')
plt.ylabel('엔트로피(bits)')
plt.xlabel('객실등급')
plt.legend(title = '성별')
# plt.show()

print('-------------------------------------------')
df = sns.load_dataset('titanic')
df = df[['age', 'embarked' ,'survived']].dropna()
# print(df.isnull().sum())
print(df)

bins = [0, 10, 20, 30, 40, 50, 60, 80]
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']
df['age-group'] = pd.cut(df['age'], bins, labels=labels)
print(df)

result2 = []
for group, subset in df.groupby('age-group'):
    print('group:', group)
    print('subset:', subset)
    counts = Counter(subset['survived'])
    print('counts:', counts)

    total_count = sum(counts.values())
    ratios = {key: value / total_count for key, value in counts.items()}
    print('ratios:', ratios)

    probabilities = [value for value in ratios.values()]
    print('probabilities:', probabilities)

    entropy = entropy_test(probabilities)
    print('entropy:', entropy)

    result2.append({'나이대':group, '엔트로피': entropy})

print(result2)
df2 = pd.DataFrame(result2)
# print(df2)

