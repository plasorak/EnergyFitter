import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import xml.etree.ElementTree as etree

def EnergyFitterCore:

    def __init__(self):
        self.Debug=False
        self.DoneMinimisation = False
        self.DoneInputDataInitialisation = False
        self.DoNormalisation = True

        self.InputFile = ""
        self.InputData = pd.DataFrame()
        
        self.RandomSeed = -1
        self.BatchSize = -1
        self.ActivationFunction = ""
        self.nNeuron = -1
        self.Minimiser = "" 
        self.LearningRate = 0
        self.nIteration = 0
        self.TrainingFraction = 0.5
        self.CrossValidationFraction = 0.25
        
        self.CrossValidationCost = []
        self.TrainCost           = []
        self.TestCost = 0
        
        self.x_cross = np.zeros(1)
        self.y_cross = np.zeros(1)
        self.x_train = np.zeros(1)
        self.y_train = np.zeros(1)
        self.x_test  = np.zeros(1)
        self.y_test  = np.zeros(1)
        
        self.x_mean   = np.zeros(4)
        self.x_stddev = np.zeros(4)
        self.y_mean   = 0
        self.y_stddev = 0
        
    def Help(self):
        message='''
This class is a wrapper of TensorFlow which is aimed at training a neural network for energy reconstruction of simple cluster used in the triggering at DUNE.
To use it:
ef=EnergyFitterCore()
ef.InputFile=\"InputData.csv\"
ef.DoNormalisation=True        # If you want to normalise you data so that they are all centred around 0 with RMS of 1 (useful since the data can have very different range, SumADC, NChan have very different values and that biases the algorithm)
ef.InitialiseInputData()
ef.RandomSeed=1234
ef.ActivationFunction=\"relu\" # can be also sigmoid for now
ef.nNeuron=100                 # any number bigger than 1
ef.Minimiser=\"adam\"          # or grad
ef.LearningRate=0.001          # a small number bigger than 0
ef.nIteration=10000            # How long you are disposed to wait
ef.Train()
ef.ExportToXML(\"SomeFile.xml\") # which can be then be fed to the triggering algorithm
'''
        print(message)
    
    def TransformToNormalised(self, x, mean, stddev):
        x_normalised = (x-mean)/stddev
        return x_normalised

    def TransformToReal(self, x, mean, rms):
        return x*rms+mean

    def NeuralNetworkOneHiddenLayer(self, x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer
    
    def InitialiseInputData(self):
        if self.InputFile=="":
            print("Input file wasn't set, do EnergyFitterCore.InputFile=\"File.csv\"")
            return

        print("Input file is"+self.InputFile)
        # Read the csv
        self.InputData = pd.read_csv(self.InputFile, sep=",")
        # Shuffle it
        self.InputData = self.InputData.sample(frac=1)

        # Define 2 pandas which contains the training data
        train_x = pd.DataFrame()
        train_x['ChanWidth'] = np.float32(self.InputData.ChanWidth)
        train_x['NHits']     = np.float32(self.InputData.NHits    )
        train_x['SumADC']    = np.float32(self.InputData.SumADC   )
        train_x['TimeWidth'] = np.float32(self.InputData.TimeWidth)
        train_x = pd.concat([train_x], axis=1)

        train_y = pd.DataFrame()
        train_y['ENu'] = np.float32(self.InputData.ENu)

        cross_cnt = floor(train_x.shape[0] * self.TrainingFraction)
        train_cnt = floor(train_x.shape[0] * (self.TrainingFraction+
                                              self.CrossValidationFraction))
                
        self.x_cross = train_x.iloc[0:cross_cnt].values
        self.y_cross = train_y.iloc[0:cross_cnt].values
        self.x_train = train_x.iloc[cross_cnt:train_cnt].values
        self.y_train = train_y.iloc[cross_cnt:train_cnt].values
        self.x_test  = train_x.iloc[train_cnt:].values
        self.y_test  = train_y.iloc[train_cnt:].values
        
        self.x_mean   = [np.mean(x_train[:,i] ) for i in range(0,4)]
        self.x_stddev = [np.std (x_train[:,i] ) for i in range(0,4)]
        self.y_mean   = np.mean(y_train)
        self.y_stddev = np.std (y_train)

        if DoNormalisation:
            self.x_train = np.float32([np.float32(TransformToNormalisedFromData(self.x_train[:,i], self.x_mean[i], self.x_stddev[i])) for i in range(0,4)])
            self.x_cross = np.float32([np.float32(TransformToNormalisedFromData(self.x_cross[:,i], self.x_mean[i], self.x_stddev[i])) for i in range(0,4)])
            self.x_test  = np.float32([np.float32(TransformToNormalisedFromData(self.x_test [:,i], self.x_mean[i], self.x_stddev[i])) for i in range(0,4)])
            self.x_train = np.transpose(self.x_train)
            self.x_cross = np.transpose(self.x_cross)
            self.x_test  = np.transpose(self.x_test )
            self.y_train = np.float32(TransformToNormalisedFromData(self.y_train, self.y_mean, self.y_stddev))
            self.y_cross = np.float32(TransformToNormalisedFromData(self.y_cross, self.y_mean, self.y_stddev))
            self.y_test  = np.float32(TransformToNormalisedFromData(self.y_test , self.y_mean, self.y_stddev))
        
        self.DoneInputDataInitialisation=True
            
    def Train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            print("Optimization for", '%d' %self.nNeuron, " neurons starting!")
            for epoch in range(self.nIteration):
                if self.Debug:
                    print("EPOCH",epoch)
                    print("biases",sess.run(biases['b1']))
                    print("outbias",sess.run(biases['out']))
                    print("weights[h1]",sess.run(weights['h1']))
                    print("weights[out]",sess.run(weights['out']))
                avg_cost = 0.0
                total_batch = int(len(self.x_train) / self.BatchSize)
                #x_batches = np.array_split(x_train, total_batch)
                #y_batches = np.array_split(y_train, total_batch)
                x_batch = self.x_train
                y_batch = self.y_train
                #for i in range(total_batch):
                #batch_x, batch_y = x_batches[i], y_batches[i]

                _, c_train, pred = sess.run([optimizer, cost, predictions_nn], feed_dict={x: x_batch, y: y_batch})

                if self.Debug:
                    print("ctrain",c_train)
                    print("pred",pred)

                c_cross, _ = sess.run([cost, predictions_nn], feed_dict={x: x_cross, y: y_cross})
                
                self.CrossValidationCost.append(c_cross)
                self.TrainCost          .append(c_train)
                
                avg_cost += c_train / total_batch
                if epoch % display_step == 0:
                    print("Epoch:", '%06d' % (epoch), "cost=", "{:.9f}".format(avg_cost))
                    previous_cost=avg_cost
            print("Optimization for", '%d' %self.nNeuron, " neurons finished!")
            DoneMinimisation=true
            
    def ExportXML(self, name):
        if not DoneMinimisation:
            print("The minimisation wasn't done, not saving the XML")
            return
        etree.Element(name)


    

        








#             cost_after_min[n_hidden]=c_cross
            
#             fig, ax = plt.subplots(figsize=(20, 12))
#             fig.suptitle("Cost")
#             ax.set_xlabel("Iteration")
#             ax.set_ylabel("Cost")
#             trainplot = plt.semilogy(range(len(cost_train_arr)), cost_train_arr, 'o-')
#             crossplot = plt.semilogy(range(len(cost_cross_arr)), cost_cross_arr, 'o-')
#             plt.legend([trainplot[0], crossplot[0]], ('Training set', 'Cross validation set'))
#             fig.savefig("relu/Cost_nhid%d.pdf" %n_hidden)
            
#             pred_cross = sess.run(predictions_nn, feed_dict={x: x_cross})
            
#             train_output[n_hidden]=pd.DataFrame()
#             print((pred_cross.shape))
#             print((y_cross.shape))
#             train_output[n_hidden]['predicted'] = np.float32(TransformToReal(pred_cross[:,0], y_mean,y_stddev))
#             train_output[n_hidden]['true']      = np.float32(TransformToReal(y_cross[:,0]   , y_mean,y_stddev))
#             fig, ax = plt.subplots(figsize=(20, 12))
#             plt.hist(x=y_cross[:,0], bins=10, range=[0,50])
#             fig.savefig("relu/Enu_Hist2.pdf")
            
#             fig, ax = plt.subplots(figsize=(20, 12))
#             plt.hist(x=train_output[n_hidden]['true'], bins=10, range=[0,50])
#             fig.savefig("relu/Enu_Hist3.pdf")
            
            
#             digitized = np.digitize(train_output[n_hidden]['true'], bins)
#             train_output[n_hidden]['true_binned'] = np.float32(digitized)
#             train_output[n_hidden]['bias'] = np.float32(pred_cross[:,0] - y_cross[:,0])
#             train_output[n_hidden]['rms']  = np.float32((pred_cross[:,0] - y_cross[:,0]) * (pred_cross[:,0] - y_cross[:,0]))
            
#             for i in range(0,train_output[n_hidden]['true'].shape[0]):
#                 if train_output[n_hidden]['predicted'][i]>100 or train_output[n_hidden]['predicted'][i]<0:
#                     print("enu  ", train_output[n_hidden]['predicted'][i])
#                     print("erec ", train_output[n_hidden]['true'][i])
                    
                    
                    
                    
                    
                    
                    
                    
                    


# debug=True
# debug=False
# sns.set(style='ticks', palette='Spectral', font_scale=1.5)

# material_palette = ["#4CAF50", "#2196F3", "#9E9E9E", "#FF9800", "#607D8B", "#9C27B0"]
# sns.set_palette(material_palette)
# rcParams['figure.figsize'] = 16, 8

# plt.xkcd();
# random_state = 42
# np.random.seed(random_state)
# tf.set_random_seed(random_state)
# pd.options.display.max_rows = 10
# pd.options.display.float_format = '{:.3f}'.format
# clustering_df = pd.read_csv("Config5.txt", sep=",")
# clustering_df=clustering_df.sample(frac=1)

# train_x = pd.DataFrame()
# train_y = pd.DataFrame()
# train_x['ChanWidth'] = np.float32(clustering_df.ChanWidth)
# train_x['NHits']     = np.float32(clustering_df.NHits)
# train_x['SumADC']    = np.float32(clustering_df.SumADC)
# train_x['TimeWidth'] = np.float32(clustering_df.TimeWidth)
# train_x = pd.concat([train_x], axis=1)
# train_y['ENu'] = np.float32(clustering_df.ENu)
# fig, ax = plt.subplots(figsize=(20, 12))

# fig.suptitle("Enu")
# ax.set_ylabel("Event")
# ax.set_xlabel("Enu [MeV]")
# plt.hist(x=clustering_df.ENu, bins=10, range=[0,50])
# fig.savefig("relu/Enu_Hist.pdf")

# train_size = 0.5
# crossval_size = 0.25
# cross_cnt = floor(train_x.shape[0] * crossval_size)
# train_cnt = floor(train_x.shape[0] * (train_size+crossval_size))


# x_cross = train_x.iloc[0:cross_cnt].values
# y_cross = train_y.iloc[0:cross_cnt].values
# x_train = train_x.iloc[cross_cnt:train_cnt].values
# y_train = train_y.iloc[cross_cnt:train_cnt].values
# x_test  = train_x.iloc[train_cnt:].values
# y_test  = train_y.iloc[train_cnt:].values

# x_mean   = [ np.mean(x_train[:,i] ) for i in range(0,4)]
# x_stddev = [ np.std (x_train[:,i] ) for i in range(0,4)]
# y_mean   = np.mean(y_train)
# y_stddev = np.std (y_train)
# print(x_test.shape)

# x_train = np.float32([ np.float32(TransformToNormalisedFromData(x_train[:,i], x_mean[i], x_stddev[i])) for i in range(0,4) ])
# x_cross = np.float32([ np.float32(TransformToNormalisedFromData(x_cross[:,i], x_mean[i], x_stddev[i])) for i in range(0,4) ])
# x_test  = np.float32([ np.float32(TransformToNormalisedFromData(x_test [:,i], x_mean[i], x_stddev[i])) for i in range(0,4) ])
# x_train = np.transpose(x_train)
# x_cross = np.transpose(x_cross)
# x_test  = np.transpose(x_test )
# y_train = np.float32(TransformToNormalisedFromData(y_train, y_mean, y_stddev))
# y_cross = np.float32(TransformToNormalisedFromData(y_cross, y_mean, y_stddev))
# y_test  = np.float32(TransformToNormalisedFromData(y_test , y_mean, y_stddev))
# print(x_test.shape)


# bins = np.linspace(0, 50, 51)
# layer_array=[10,20,30,50,100,200]
# #layer_array=[10,20,30]
# # if debug:
# #layer_array=[50]
# cost_after_min={}
# learning_rates={10:0.0001,20:0.0001,30:0.0001,50:0.0001,100:0.0001,200:0.0001}
# number_iter={10:2000,20:4000,30:6000,50:10000,100:20000,200:20000}
# previous_cost=10000000
# train_output={}

# for n_hidden in layer_array:
#     n_input = train_x.shape[1]
#     n_classes = 1

#     weights = {
#         'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
#         'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
#     }
    
#     biases = {
#         'b1': tf.Variable(tf.random_normal([n_hidden])),
#         'out': tf.Variable(tf.random_normal([n_classes]))
#     }
    
#     #keep_prob = tf.placeholder("float")
    
#     training_epochs = number_iter[n_hidden]
#     if debug:
#         training_epochs=1;
#     cost_train_arr=[]
#     cost_cross_arr=[]
#     display_step = 100
#     batch_size = 100
    
#     x = tf.placeholder("float", [None, n_input])
#     y = tf.placeholder("float", [None, n_classes])
        
#     predictions_nn = NeuralNetworkOneHiddenLayer(x, weights, biases)
#     cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=predictions_nn, labels=y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rates[n_hidden]).minimize(cost)
    
        
# for n_hidden, output in train_output.items():
#     fig, ax = plt.subplots(figsize=(20, 12))
#     fig.suptitle("Smearing")
#     ax.set_xlabel('E True [MeV]')
#     ax.set_ylabel('E Reco [MeV]')
#     plt.hist2d(output['true'],output['predicted'], bins=100, range=[[0,100],[0,100]],norm=colors.LogNorm())
#     plt.colorbar()
#     plt.grid(True)
#     fig.savefig("relu/Smearing_%d.pdf"%n_hidden)

    
# for n_hidden, output in train_output.items():
#     fig, ax = plt.subplots(figsize=(20, 12))
#     fig.suptitle("Smearing")
#     ax.set_xlabel('E True [MeV]')
#     ax.set_ylabel('E Reco [MeV]')
#     sns.regplot(x=output['true_binned'],y=output['predicted'],x_bins=bins, fit_reg=None)
#     plt.grid(True)
#     fig.savefig("relu/Profile_%d.pdf"%n_hidden)

# fig, (axs1,axs2) = plt.subplots(2, 1, sharex='col',figsize=(20, 12))
# fig.subplots_adjust(hspace=0)
# fig.suptitle("Resolution and Bias")
# plt.subplot(2, 1, 1)
# i=0
# for n_hidden, output in train_output.items():
#     sns.regplot(x=output['true_binned'],y=output['rms'], x_bins=bins, fit_reg=None,color=material_palette[i],label="%d hidden layers"%n_hidden)
#     i+=1
# plt.grid(True)
# plt.ylabel('Relative Error [MeV]')
# plt.legend()
# plt.subplot(2, 1, 2)
# i=0
# for n_hidden, output in  train_output.items():
#     sns.regplot(x=output['true_binned'],y=output['bias'], x_bins=bins, fit_reg=None,color=material_palette[i])
#     i+=1
# plt.grid(True)
# plt.xlabel('E True [MeV]')
# plt.ylabel('Bias (Reco-True) [MeV]')
# fig.savefig("relu/ErrorAndBias.pdf")
       
# fig, ax = plt.subplots(figsize=(20, 12))
# fig.suptitle("Cost")
# ax.set_xlabel("Number of hidden layers")
# ax.set_ylabel("Cost")
# keys = np.fromiter(cost_after_min.keys(), dtype=float)
# vals = np.fromiter(cost_after_min.values(), dtype=float)
# plt.plot(keys,vals, 'o-')
# fig.savefig("relu/CostsVSLayers.pdf")



