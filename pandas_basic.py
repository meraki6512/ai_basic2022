# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:43:56 2022

@author: 82103
"""

import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()

df = pd.DataFrame(data['data'], columns = data.feature_names)

#1. 
df.head()
df.head(-146)

#2.
df.hist()

#3.
df.idxmax()

#4.
df.index
df.columns

#5. 
df.dtypes

#6.
df.empty

#7.
df.iat[2,2]

#8.
df.iloc[0]
df.iloc[0,1]
df.iloc[[0,1]]
df.iloc[:1]

#9.
df.info(verbose=False)

#10.
df.select_dtypes(include=['float64'])

#11.
df.values
df.to_numpy()

#12.
df.axes

#13. 
df.ndim

#14. 
df.size

#15.
df.shape 

#16
df.memory_usage(index=False)

#17
df.flags
df2 = df.set_flags(allows_duplicate_labels=False)
df2.flags.allows_duplicate_labels

#18
df.astype('category').dtypes
df.astype({'sepal length (cm)': 'int64'}).dtypes

#19
df.convert_dtypes()
df.dtypes

#20
deep = df.copy()
shallow = df.copy(deep=False)
df.iat[0,1]
df.iat[0,1] = 9.0
deep.iat[0,1]
shallow.iat[0,1]

#21
df.__iter__()

#22
for label, content in df.items():
    print(f'label: {label}')
    print(f'content: {content}', sep='\n')

#23
df.keys()

#24
df.xs(2)

#25
df.get(["sepal length (cm)", "petal length (cm)"], default="default_value")

#26
df.head()
df.add(10).head()
df.sub(10).head()
df.mul(10).head()
df.div(10).head()
df.mod(5).head()

#27
df2 = df.add(10)
take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
df.combine(df2, take_smaller, fill_value=-5)


#28
import numpy as np
df.apply(np.sqrt)

#29
df.agg(['sum', 'min'])
df.agg(['min', 'max'])

#30
df.head(10)
df.clip(2, 5)










