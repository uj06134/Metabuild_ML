import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import font_manager

font_location = 'c:/Windows/fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

data = {
    '날짜': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05'],
    '기온(°C)': [30.5, 32.0, 33.5, 31.0, 29.5],
    '여행자 수': [150, 200, 250, 180, 120]
}

corr = np.corrcoef((data['기온(°C)'], data['여행자 수']))
mask = np.triu(np.ones_like(corr, dtype=bool))
print(mask)

sns.heatmap(corr, annot=True, cmap="gray", mask=mask, fmt=".2f")
plt.title("기온과 여행자 수의 상관관계")
plt.show()