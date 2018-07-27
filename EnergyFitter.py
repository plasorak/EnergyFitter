import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams


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

print(clustering_df.describe())
sns_plot = sns.pairplot(clustering_df[['ChanWidth', 'NHits', 'SumADC', 'TimeWidth', 'ENu']]);
sns_plot.savefig("AllDistribs.pdf")

corr_mat = clustering_df[['ChanWidth', 'NHits', 'SumADC', 'TimeWidth', 'ENu']].corr() 
fig, ax = plt.subplots(figsize=(20, 12))
fig.suptitle("Correlations")
sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax)
fig.savefig("Correlations.pdf")


train_x=pd.DataFrame()
train_y=pd.DataFrame()
train_x['ChanWidth'] = np.float32(clustering_df.ChanWidth)
train_x['NHits'] = np.float32(clustering_df.NHits)
train_x['SumADC'] = np.float32(clustering_df.SumADC)
train_x['TimeWidth'] = np.float32(clustering_df.TimeWidth)
train_x = pd.concat([train_x], axis=1)
train_y['ENu'] = np.float32(clustering_df.ENu)


train_size = 0.5
crossval_size = 0.25
train_cnt = floor(train_x.shape[0] * train_size)
cross_cnt = floor(train_x.shape[0] * (train_size+crossval_size))
x_train = train_x.iloc[0:train_cnt].values
y_train = train_y.iloc[0:train_cnt].values
x_cross = train_x.iloc[train_cnt:cross_cnt].values
y_cross = train_y.iloc[train_cnt:cross_cnt].values
x_test  = train_x.iloc[cross_cnt:].values
y_test  = train_y.iloc[cross_cnt:].values



def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

#layer_array=[10,20,30,50,100]
#layer_array=[10,30,50]
layer_array=[10]
cost_after_min={}
learning_rates={10:0.0001,20:0.0001,30:0.0001,50:0.0001,100:0.0001}
number_iter={10:200,20:40000,30:60000,50:100000,100:200000}
previous_cost=10000000

for n_hidden in layer_array:
    n_input = train_x.shape[1]
    print(n_input)
    n_classes = 1

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    keep_prob = tf.placeholder("float")
    
    training_epochs = number_iter[n_hidden]
    cost_train_arr=[]
    cost_cross_arr=[]
    display_step = 100
    batch_size = 100
    
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
        
    predictions_nn = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=predictions_nn, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rates[n_hidden]).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        print("Optimization for", '%d' %n_hidden, " hidden layers starting!")
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            #x_batches = np.array_split(x_train, total_batch)
            #y_batches = np.array_split(y_train, total_batch)
            x_batch = x_train
            y_batch = y_train
            #for i in range(total_batch):
            #batch_x, batch_y = x_batches[i], y_batches[i]
            _, c_train, pred = sess.run([optimizer, cost, predictions_nn], feed_dict={x: x_batch, y: y_batch})
            c_cross, _ = sess.run([cost, predictions_nn], feed_dict={x: x_cross, y: y_cross})
            # multilayer_perceptron(x_batch, weights, biases)
            cost_train_arr.append(c_train)
            cost_cross_arr.append(c_cross)
            avg_cost += c_train / total_batch
            if epoch % display_step == 0:
                if abs(previous_cost-avg_cost)/avg_cost<0.0001:
                    break
                print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                previous_cost=avg_cost
        print("Optimization for", '%d' %n_hidden, " hidden layers finished!")
        cost_after_min[n_hidden]=c_cross
        
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.suptitle("Cost")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        plt.plot(range(len(cost_train_arr)),cost_train_arr, 'o-', range(len(cost_cross_arr)), cost_cross_arr, 'b-')
        fig.savefig("Cost_nhid%d.pdf" %n_hidden)
        pred_cross = sess.run(predictions_nn, feed_dict={x: x_train, weights: ,biases: })
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.suptitle("Smearing")
        ax.set_xlabel('E True [MeV]')
        ax.set_ylabel('E Reco [MeV]')
        plt.plot(y_train,pred_cross, 'o')
        fig.savefig("Results_%d.pdf"%n_hidden)
        
        


        
fig, ax = plt.subplots(figsize=(20, 12))
fig.suptitle("Cost")
ax.set_xlabel("Number of hidden layers")
ax.set_ylabel("Cost")
keys = np.fromiter(cost_after_min.keys(), dtype=float)
vals = np.fromiter(cost_after_min.values(), dtype=float)
print(type(keys))
print(type(vals))
plt.plot(keys,vals, 'o-')
fig.savefig("CostsVSLayers.pdf")
