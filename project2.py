#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

f= open('上证指数.csv')
data = pd.read_csv(u'上证指数.csv', delim_whitespace = False, encoding= 'gbk')
data.info()
plt.plot(data[u'最高价'].values[:500])
y = gaussian_filter1d(data[u'最高价'].values[:500], sigma=10)
plt.plot(y)
plt.show()
