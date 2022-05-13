# %%
# import libraries
import numpy as np
import pandas as pd
import pandas_datareader.data as web

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-whitegrid')

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# %%
# load stock data
df = web.DataReader('AAPL', 'yahoo', start='2020-01-01', end='2021-12-31')
# %%
# label data
df['label'] = (df['Close'].shift(-1) >= df['Close']).astype(int)

# %%
# create features
def ts_3point_shape(left_point,mid_point,right_point):
    slope_left = mid_point - left_point
    slope_right = right_point - mid_point
    
    # down slope + lower down
    if slope_left < 0 and slope_left < slope_right and slope_right < 0:
        return 1
    
    # up slope + lower up
    if slope_left > 0 and slope_left > slope_right and slope_right > 0:
        return 2 
    
    # up slope + greater up
    if slope_left > 0 and slope_left < slope_right and slope_right > 0:
        return 3
    
    # down slope + greater down
    if slope_left < 0 and slope_left > slope_right and slope_right < 0:
        return 4
    
    # down slope + up slope
    if slope_left < 0 and slope_left < slope_right and slope_right > 0:
        return 5
    
    # up slope + down slope
    if slope_left > 0 and slope_left > slope_right and slope_right < 0:
        return 6
    
    return 0
    
df['shape_feature'] = df['Close'].rolling(window=3).apply(lambda x : ts_3point_shape(x[0],x[1],x[2]))
# %%
print('Total Number of Rows {}'.format(df.shape[0]))
df.groupby('shape_feature').count()['label'].plot.bar()
# %%
print('Total Average of Label {}'.format(df['label'].mean()))
df.groupby('shape_feature').mean()['label'].plot.bar()
# %%
# train test split for model
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42,shuffle=False)

X_train = df_train[['shape_feature']]
y_train = df_train['label']

X_test = df_test[['shape_feature']]
y_test = df_test['label']

# one hot encoding of shape feature
one_hot_encoder = OneHotEncoder()
X_train = one_hot_encoder.fit_transform(X_train)
X_test = one_hot_encoder.transform(X_test)

# %%
model = RidgeClassifier()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print('Train Score: {:%} - Test Score: {:%}'.format(train_score, test_score))
print('Average of Label Train:{:%} - Test:{:%}'.format(df_train['label'].mean(),df_test['label'].mean()))

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
predictions = model.predict(X_test)
cm = confusion_matrix(df_test['label'], predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
# %%
