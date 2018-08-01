import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams
from matplotlib.backends.backend_pdf import PdfPages

debug=True
debug=False
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
print(x_test.shape)

x_train = np.float32([ np.float32(TransformToNormalisedFromData(x_train[:,i], x_mean[i], x_stddev[i])) for i in range(0,4) ])
x_cross = np.float32([ np.float32(TransformToNormalisedFromData(x_cross[:,i], x_mean[i], x_stddev[i])) for i in range(0,4) ])
x_test  = np.float32([ np.float32(TransformToNormalisedFromData(x_test [:,i], x_mean[i], x_stddev[i])) for i in range(0,4) ])
x_train = np.transpose(x_train)
x_cross = np.transpose(x_cross)
x_test  = np.transpose(x_test )
y_train = np.float32(TransformToNormalisedFromData(y_train, y_mean, y_stddev))
y_cross = np.float32(TransformToNormalisedFromData(y_cross, y_mean, y_stddev))
y_test  = np.float32(TransformToNormalisedFromData(y_test , y_mean, y_stddev))
print(x_test.shape)

def NeuralNetworkOneHiddenLayer(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

bins = np.linspace(0, 50, 51)
layer_array=[10,20,30,50,100,200]
#  layer_array=[10,20,30]
# if debug:
layer_array=[50]
cost_after_min={}
learning_rates={10:0.0001,20:0.0001,30:0.0001,50:0.0001,100:0.0001,200:0.0001}
number_iter={10:20000,20:40000,30:60000,50:1000,100:200000,200:200000}
previous_cost=10000000
train_output={}

for n_hidden in layer_array:
    n_input = train_x.shape[1]
    n_classes = 1

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    #keep_prob = tf.placeholder("float")
    
    training_epochs = number_iter[n_hidden]
    if debug:
        training_epochs=1;
    cost_train_arr=[]
    cost_cross_arr=[]
    display_step = 100
    batch_size = 100
    
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
        
    predictions_nn = NeuralNetworkOneHiddenLayer(x, weights, biases)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=predictions_nn, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rates[n_hidden]).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        print("Optimization for", '%d' %n_hidden, " hidden layers starting!")
        for epoch in range(training_epochs):
            if debug:
                print("EPOCH",epoch)
                print("biases",sess.run(biases['b1']))
                print("outbias",sess.run(biases['out']))
                print("weights[h1]",sess.run(weights['h1']))
                print("weights[out]",sess.run(weights['out']))
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            #x_batches = np.array_split(x_train, total_batch)
            #y_batches = np.array_split(y_train, total_batch)
            x_batch = x_train
            y_batch = y_train
            #for i in range(total_batch):
            #batch_x, batch_y = x_batches[i], y_batches[i]
            _, c_train, pred = sess.run([optimizer, cost, predictions_nn], feed_dict={x: x_batch, y: y_batch})
            if debug:
                print("ctrain",c_train)
                print("pred",pred)
            c_cross, _ = sess.run([cost, predictions_nn], feed_dict={x: x_cross, y: y_cross})
            # multilayer_perceptron(x_batch, weights, biases)
            cost_train_arr.append(c_train)
            cost_cross_arr.append(c_cross)
            avg_cost += c_train / total_batch
            if epoch % display_step == 0:
                # if abs(previous_cost-avg_cost)/avg_cost<0.0001:
                #     break
                print("Epoch:", '%06d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                previous_cost=avg_cost
        print("Optimization for", '%d' %n_hidden, " hidden layers finished!")
        cost_after_min[n_hidden]=c_cross
        
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.suptitle("Cost")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        trainplot = plt.semilogy(range(len(cost_train_arr)), cost_train_arr, 'o-')
        crossplot = plt.semilogy(range(len(cost_cross_arr)), cost_cross_arr, 'o-')
        plt.legend([trainplot[0], crossplot[0]], ('Training set', 'Cross validation set'))
        fig.savefig("Cost_nhid%d.pdf" %n_hidden)

        pred_cross = sess.run(predictions_nn, feed_dict={x: x_cross})

        train_output[n_hidden]=pd.DataFrame()
        print((pred_cross.shape))
        print((y_cross.shape))
        train_output[n_hidden]['predicted'] = np.float32(TransformToReal(pred_cross[:,0], y_mean,y_stddev))
        train_output[n_hidden]['true']      = np.float32(TransformToReal(y_cross[:,0]   , y_mean,y_stddev))
        fig, ax = plt.subplots(figsize=(20, 12))
        plt.hist(x=y_cross[:,0], bins=10, range=[0,50])
        fig.savefig("Enu_Hist2.pdf")

        fig, ax = plt.subplots(figsize=(20, 12))
        plt.hist(x=train_output[n_hidden]['true'], bins=10, range=[0,50])
        fig.savefig("Enu_Hist3.pdf")


        digitized = np.digitize(train_output[n_hidden]['true'], bins)
        train_output[n_hidden]['true_binned'] = np.float32(digitized)
        train_output[n_hidden]['bias'] = np.float32(pred_cross[:,0] - y_cross[:,0])
        train_output[n_hidden]['rms']  = np.float32((pred_cross[:,0] - y_cross[:,0]) * (pred_cross[:,0] - y_cross[:,0]))
        
        for i in range(0,train_output[n_hidden]['true'].shape[0]):
            if train_output[n_hidden]['predicted'][i]>100 or train_output[n_hidden]['predicted'][i]<0:
                print("enu  ", train_output[n_hidden]['predicted'][i])
                print("erec ", train_output[n_hidden]['true'][i])
        
for n_hidden, output in train_output.items():
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.suptitle("Smearing")
    ax.set_xlabel('E True [MeV]')
    ax.set_ylabel('E Reco [MeV]')
    plt.hist2d(output['true'],output['predicted'], bins=100, range=[[0,100],[0,100]],norm=colors.LogNorm())
    plt.colorbar()
    plt.grid(True)
    fig.savefig("Smearing_%d.pdf"%n_hidden)

    
for n_hidden, output in train_output.items():
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.suptitle("Smearing")
    ax.set_xlabel('E True [MeV]')
    ax.set_ylabel('E Reco [MeV]')
    sns.regplot(x=output['true_binned'],y=output['predicted'],x_bins=bins, fit_reg=None)
    plt.grid(True)
    fig.savefig("Profile_%d.pdf"%n_hidden)

fig, (axs1,axs2) = plt.subplots(2, 1, sharex='col',figsize=(20, 12))
fig.subplots_adjust(hspace=0)
fig.suptitle("Resolution and Bias")
plt.subplot(2, 1, 1)
i=0
for n_hidden, output in train_output.items():
    sns.regplot(x=output['true_binned'],y=output['rms'], x_bins=bins, fit_reg=None,color=material_palette[i],label="%d hidden layers"%n_hidden)
    i+=1
plt.grid(True)
plt.ylabel('Relative Error [MeV]')
plt.legend()
plt.subplot(2, 1, 2)
i=0
for n_hidden, output in  train_output.items():
    sns.regplot(x=output['true_binned'],y=output['bias'], x_bins=bins, fit_reg=None,color=material_palette[i])
    i+=1
plt.grid(True)
plt.xlabel('E True [MeV]')
plt.ylabel('Bias (Reco-True) [MeV]')
fig.savefig("ErrorAndBias.pdf")
       
fig, ax = plt.subplots(figsize=(20, 12))
fig.suptitle("Cost")
ax.set_xlabel("Number of hidden layers")
ax.set_ylabel("Cost")
keys = np.fromiter(cost_after_min.keys(), dtype=float)
vals = np.fromiter(cost_after_min.values(), dtype=float)
plt.plot(keys,vals, 'o-')
fig.savefig("CostsVSLayers.pdf")
