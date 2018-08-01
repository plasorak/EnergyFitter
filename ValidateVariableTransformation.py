import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style='ticks', palette='Spectral', font_scale=1.5)

material_palette = ["#4CAF50", "#2196F3", "#9E9E9E", "#FF9800", "#607D8B", "#9C27B0"]
sns.set_palette(material_palette)
rcParams['figure.figsize'] = 16, 8

plt.xkcd();
random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format
clustering_df = pd.read_csv("Config5.txt", sep=",")

# print(clustering_df.describe())
# sns_plot = sns.pairplot(clustering_df[['ChanWidth', 'NHits', 'SumADC', 'TimeWidth', 'ENu']]);
# sns_plot.savefig("AllDistribs.pdf")

# corr_mat = clustering_df[['ChanWidth', 'NHits', 'SumADC', 'TimeWidth', 'ENu']].corr() 
# fig, ax = plt.subplots(figsize=(20, 12))
# fig.suptitle("Correlations")
# sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax)
# fig.savefig("Correlations.pdf")

clustering_df=clustering_df.sample(frac=1)
def TransformToNormalised(x):
    stddev=np.std(x)
    mean=np.mean(x)
    x_normalised = (x-mean)/stddev
    return (x_normalised,mean,stddev)

def TransformToNormalisedFromData(x,mean,stddev):
    x_normalised = (x-mean)/stddev
    return x_normalised

def TransformToReal(x, mean, rms):
    return x*rms+mean

train_x = pd.DataFrame()
train_y = pd.DataFrame()
train_x['ChanWidth'] = np.float32(clustering_df.ChanWidth)
train_x['NHits']     = np.float32(clustering_df.NHits)
train_x['SumADC']    = np.float32(clustering_df.SumADC)
train_x['TimeWidth'] = np.float32(clustering_df.TimeWidth)
train_x = pd.concat([train_x], axis=1)
train_y['ENu'] = np.float32(clustering_df.ENu)
fig, ax = plt.subplots(figsize=(20, 12))

fig.suptitle("Enu")
ax.set_ylabel("Event")
ax.set_xlabel("Enu [MeV]")
plt.hist(x=clustering_df.ENu, bins=10, range=[0,50])
fig.savefig("Enu_Hist.pdf")

train_size = 0.5
crossval_size = 0.25
cross_cnt = floor(train_x.shape[0] * crossval_size)
train_cnt = floor(train_x.shape[0] * (train_size+crossval_size))


x_cross = train_x.iloc[0:cross_cnt].values
y_cross = train_y.iloc[0:cross_cnt].values
x_train = train_x.iloc[cross_cnt:train_cnt].values
y_train = train_y.iloc[cross_cnt:train_cnt].values
x_test  = train_x.iloc[train_cnt:].values
y_test  = train_y.iloc[train_cnt:].values

x_mean   = [ np.mean(x_train[:,i] ) for i in range(0,4)]
x_stddev = [ np.std (x_train[:,i] ) for i in range(0,4)]
y_mean   = np.mean(y_train)
y_stddev = np.std (y_train)
 
x_train = [ TransformToNormalisedFromData(x_train[:,i], x_mean[i], x_stddev[i]) for i in range(0,4) ]
x_cross = [ TransformToNormalisedFromData(x_cross[:,i], x_mean[i], x_stddev[i]) for i in range(0,4) ]
x_test  = [ TransformToNormalisedFromData(x_test [:,i], x_mean[i], x_stddev[i]) for i in range(0,4) ]
y_train = TransformToNormalisedFromData(y_train, y_mean, y_stddev)
y_cross = TransformToNormalisedFromData(y_cross, y_mean, y_stddev)
y_test  = TransformToNormalisedFromData(y_test , y_mean, y_stddev)

x_cross_copy = train_x.iloc[0:cross_cnt].values
y_cross_copy = train_y.iloc[0:cross_cnt].values
x_train_copy = train_x.iloc[cross_cnt:train_cnt].values
y_train_copy = train_y.iloc[cross_cnt:train_cnt].values
x_test_copy  = train_x.iloc[train_cnt:].values
y_test_copy  = train_y.iloc[train_cnt:].values


x_train3 = [ TransformToReal(x_train[i], x_mean[i], x_stddev[i]) for i in range(0,4) ]
x_cross3 = [ TransformToReal(x_cross[i], x_mean[i], x_stddev[i]) for i in range(0,4) ]
x_test3  = [ TransformToReal(x_test [i], x_mean[i], x_stddev[i]) for i in range(0,4) ]
print(y_train.shape)
y_train3 = TransformToReal(y_train,y_mean,y_stddev)
y_cross3 = TransformToReal(y_cross,y_mean,y_stddev)
y_test3  = TransformToReal(y_test ,y_mean,y_stddev)

raw=[x_train_copy[:,0],
     x_train_copy[:,1],
     x_train_copy[:,2],
     x_train_copy[:,3],
     x_cross_copy[:,0],
     x_cross_copy[:,1],
     x_cross_copy[:,2],
     x_cross_copy[:,3],
     x_test_copy [:,0],
     x_test_copy [:,1],
     x_test_copy [:,2],
     x_test_copy [:,3],
     y_train_copy,
     y_cross_copy,
     y_test_copy]

norm=[x_train[0],
      x_train[1],
      x_train[2],
      x_train[3],
      x_cross[0],
      x_cross[1],
      x_cross[2],
      x_cross[3],
      x_test [0],
      x_test [1],
      x_test [2],
      x_test [3],
      y_train,     
      y_cross,     
      y_test]
resurect=[x_train3[0],
          x_train3[1],
          x_train3[2],
          x_train3[3],
          x_cross3[0],
          x_cross3[1],
          x_cross3[2],
          x_cross3[3],
          x_test3 [0],
          x_test3 [1],
          x_test3 [2],
          x_test3 [3],
          y_train3,     
          y_cross3,     
          y_test3]

title=["x_train0",
       "x_train1",
       "x_train2",
       "x_train3",
       "x_cross0",
       "x_cross1",
       "x_cross2",
       "x_cross3",
       "x_test0",
       "x_test1",
       "x_test2",
       "x_test3",
       "y_train",     
       "y_cross",     
       "y_test"]
plt.close()

fig, ax = plt.subplots(figsize=(20, 12))
with PdfPages('validation_before_transform.pdf') as pdf:
    for index in range(0,15):
        plt.hist(x=raw[index], bins=20, range=[0,50])
        plt.title('Raw '+title[index])
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

with PdfPages('validation_after_transform.pdf') as pdf:
    for index in range(0,15):
        plt.hist(x=norm[index], bins=20, range=[-5,5])
        plt.title('Norm '+title[index])
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

with PdfPages('validation_after_resurection.pdf') as pdf:
    for index in range(0,15):
        plt.hist(x=resurect[index], bins=20, range=[0,50])
        plt.title('resurec '+title[index])
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
with PdfPages('validation_scatt.pdf') as pdf:
    for index in range(0,15):
        plt.scatter(x=raw[index],y=resurect[index])
        plt.title('scatt '+title[index])
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
