import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from ROOT import TXMLEngine as xml


class EnergyFitterCore:
    """This class is a wrapper of TensorFlow which is aimed at training a neural network for energy reconstruction of simple cluster used in the triggering at DUNE.
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
    ... Or you can inherit from it and initialise these variables as in EnergyFitter_sig.py
    """

    def __init__(self):
        self.Debug=False
        self.DoneMinimisation = False
        self.DoneInputDataInitialisation = False
        self.DoNormalisation = True
        self.DisplayStep=100
        self.InputFile = ""
        self.InputData = pd.DataFrame()
        
        self.RandomSeed = -1
        self.BatchSize = 1
        self.ActivationFunction = ""
        self.nNeuron = -1
        self.Minimiser = "" 
        self.LearningRate = 0
        self.nIteration = 0
        self.TrainingFraction = 0.5
        self.CrossValidationFraction = 0.25
        
        self.CrossValidationCost = []
        self.TrainCost = []
        self.TestCost = 0
        
        self.x_cross = np.zeros(1)
        self.y_cross = np.zeros(1)
        self.x_train = np.zeros(1)
        self.y_train = np.zeros(1)
        self.x_test  = np.zeros(1)
        self.y_test  = np.zeros(1)
        
        self.x_mean   = np.zeros(4)
        self.x_stddev = np.ones(4)
        self.y_mean   = 0
        self.y_stddev = 1
        self.weights = {}
        self.bias    = {}

        self.finalweights = {}
        self.finalbias    = {}
        
    def TransformToNormalised(self, x, mean, stddev):
        x_normalised = (x-mean)/stddev
        return x_normalised

    def TransformToReal(self, x, mean, rms):
        return x*rms+mean

    def Predictor(self, x, weights, biases):
        layer_1 = tf.matmul(x, weights['h1'])
        layer_1 = tf.add(layer_1, biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        out_layer = tf.matmul(layer_1, weights['out'])
        out_layer = out_layer + biases['out']
        return out_layer
    
    def InitialiseInputData(self):
        if self.InputFile=="":
            print("Input file wasn't set, do EnergyFitterCore.InputFile=\"File.csv\"")
            return

        print("Input file is "+self.InputFile)
        # Read the csv
        self.InputData = pd.read_csv(self.InputFile, sep=",")
        # Shuffle it
        self.InputData = self.InputData.sample(frac=1)
        self.InputData = self.InputData[self.InputData['Type']==1]

        # Define 2 pandas which contains the training data
        train_x = pd.DataFrame()
        train_x['ChanWidth'] = np.float32(self.InputData.ChanWidth)
        train_x['NHits']     = np.float32(self.InputData.NHits    )
        train_x['SumADC']    = np.float32(self.InputData.SumADC   )
        train_x['TimeWidth'] = np.float32(self.InputData.TimeWidth)
        train_x = pd.concat([train_x], axis=1)

        train_y = pd.DataFrame()
        train_y['ENu'] = np.float32(self.InputData.ENu)

        cross_cnt = floor(train_x.shape[0] * self.CrossValidationFraction)
        train_cnt = floor(train_x.shape[0] * (self.TrainingFraction+
                                              self.CrossValidationFraction))
                
        self.x_cross = train_x.iloc[0:cross_cnt].values
        self.y_cross = train_y.iloc[0:cross_cnt].values
        self.x_train = train_x.iloc[cross_cnt:train_cnt].values
        self.y_train = train_y.iloc[cross_cnt:train_cnt].values
        self.x_test  = train_x.iloc[train_cnt:].values
        self.y_test  = train_y.iloc[train_cnt:].values
        
        self.x_mean   = [np.mean(self.x_train[:,i] ) for i in range(0,4)]
        self.x_stddev = [np.std (self.x_train[:,i] ) for i in range(0,4)]
        self.y_mean   = np.mean(self.y_train)
        self.y_stddev = np.std (self.y_train)

        if self.DoNormalisation:
            print("Renormalising the data")
            for i in range(0,4):
                print(i, self.x_mean[i])
                print(i, self.x_stddev[i])
            self.x_train = np.float32([np.float32(self.TransformToNormalised(self.x_train[:,i], self.x_mean[i], self.x_stddev[i])) for i in range(0,4)])
            self.x_cross = np.float32([np.float32(self.TransformToNormalised(self.x_cross[:,i], self.x_mean[i], self.x_stddev[i])) for i in range(0,4)])
            self.x_test  = np.float32([np.float32(self.TransformToNormalised(self.x_test [:,i], self.x_mean[i], self.x_stddev[i])) for i in range(0,4)])
            self.x_train = np.transpose(self.x_train)
            self.x_cross = np.transpose(self.x_cross)
            self.x_test  = np.transpose(self.x_test )
            self.y_train = np.float32(self.TransformToNormalised(self.y_train, self.y_mean, self.y_stddev))
            self.y_cross = np.float32(self.TransformToNormalised(self.y_cross, self.y_mean, self.y_stddev))
            self.y_test  = np.float32(self.TransformToNormalised(self.y_test , self.y_mean, self.y_stddev))
            
        print("x_cross.shape",self.x_cross.shape)
        print("y_cross.shape",self.y_cross.shape)
        print("x_train.shape",self.x_train.shape)
        print("y_train.shape",self.y_train.shape)
        print("x_test .shape",self.x_test .shape)
        print("y_test .shape",self.y_test .shape)
            
        print("Finished instantiating the data")
        self.DoneInputDataInitialisation=True









        
    def Train(self):
        n_input = self.x_train.shape[1]
        print ("n_input",n_input)
        n_classes = 1
        
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, self.nNeuron])),
            'out': tf.Variable(tf.random_normal([self.nNeuron, n_classes]))
        }
        
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.nNeuron])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        
        
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        
        predictions = self.Predictor(x, self.weights, self.biases)
        cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=predictions, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            print("Optimization for", '%d' %self.nNeuron, " neurons starting!")

            for epoch in range(self.nIteration):
                if self.Debug:
                    if epoch>=1:
                        break
                    print("EPOCH",epoch)
                    print("biases ",       sess.run(self.biases['b1']))
                    print("outbias ",      sess.run(self.biases['out']))
                    print("weights[h1] ",  sess.run(self.weights['h1']))
                    print("weights[out] ", sess.run(self.weights['out']))
                avg_cost = 0.0
                total_batch = int(len(self.x_train) / self.BatchSize)
                #x_batches = np.array_split(x_train, total_batch)
                #y_batches = np.array_split(y_train, total_batch)
                x_batch = self.x_train
                y_batch = self.y_train
                #for i in range(total_batch):
                #batch_x, batch_y = x_batches[i], y_batches[i]

                _, c_train, pred = sess.run([optimizer, cost, predictions], feed_dict={x: x_batch, y: y_batch})

                if self.Debug:
                    print("ctrain ", c_train)
                    print("pred ",   pred)
                    print("len ctrain ", c_train.shape)
                    print("len pred ",   pred.shape)

                c_cross, _ = sess.run([cost, predictions], feed_dict={x: self.x_cross, y: self.y_cross})
                
                
                self.CrossValidationCost.append(c_cross)
                self.TrainCost          .append(c_train)
                avg_cost += c_train / total_batch
                
                if epoch % self.DisplayStep == 0 or self.Debug:
                    print("Epoch:", '%06d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

            self.finalweights['h1']  = sess.run(self.weights['h1']) 
            self.finalweights['out'] = sess.run(self.weights['out'])
            self.finalbias   ['b1']  = sess.run(self.biases['b1'])
            self.finalbias   ['out'] = sess.run(self.biases['out'])
            print("Optimization for", '%d' %self.nNeuron, " neurons finished!")
            # x_testing = np.transpose(np.float32(range(1,4)))
            x_testing = np.float32([8,7,1000,1])
            x_testing = np.float32(self.TransformToNormalised(x_testing,
                                                              np.float32(self.x_mean),
                                                              np.float32(self.x_stddev)))
            
            x_testing = [x_testing]
            y_testing = np.float32(range(0,1))
            y_testing = [y_testing]
            layer_1 = np.matmul(x_testing,self.finalweights['h1'])
            layer_1 += self.finalbias['b1']
            for i in range(0,100):
                layer_1[0,i] = max(0.,layer_1[0,i])
            out_layer = np.matmul(layer_1,self.finalweights['out'])
            out_layer += self.finalbias['out']
            _, pred = sess.run([cost, predictions], feed_dict={x: x_testing, y: y_testing})

            print("Predictions for test cluster (unnormalised):" , pred)
            pred = self.TransformToReal(pred,self.y_mean,self.y_stddev)
            print("Predictions for test cluster:" , pred)
            self.DoneMinimisation=True








            
    def ExportXML(self, name):
        if not self.DoneMinimisation:
            print("The minimisation wasn't done, not saving the XML")
            return
        filexml = xml()
        mainnode = filexml.NewChild(0, 0, "main");
        param = filexml.NewChild(mainnode, 0, "Param")

        filexml.NewChild(param, 0, "nNeuron",    str(self.nNeuron));    
        filexml.NewChild(param, 0, "Activation", self.ActivationFunction);    

        indata = filexml.NewChild(mainnode, 0, "InputData")
        i=0
        for v in self.x_mean:
            filexml.NewChild(indata, 0, "Mean_"+str(i),   str('%25.20f' % (v)))
            i+=1

        i=0
        for v in self.x_stddev:
            filexml.NewChild(indata, 0, "StdDev_"+str(i), str('%25.20f' % (v)))
            i+=1
        
        outdata = filexml.NewChild(mainnode, 0, "OutputData")
        filexml.NewChild(outdata, 0, "Mean",   str('%25.20f' % (self.y_mean  )))    
        filexml.NewChild(outdata, 0, "StdDev", str('%25.20f' % (self.y_stddev)))    
        
        bias   = filexml.NewChild(mainnode, 0, "Bias");
        weight = filexml.NewChild(mainnode, 0, "Weight");

        weight_h1  = filexml.NewChild(weight, 0, "h1");
        weight_out = filexml.NewChild(weight, 0, "out");

        bias_b1  = filexml.NewChild(bias, 0, "b1");
        bias_out = filexml.NewChild(bias, 0, "out");
        print("finalweights['h1'] .shape", self.finalweights['h1'] .shape)
        print("finalweights['out'].shape", self.finalweights['out'].shape)
        print("finalbias   ['b1'] .shape", self.finalbias   ['b1'] .shape)
        print("finalbias   ['out'].shape", self.finalbias   ['out'].shape)

        i=0;
        for v in self.finalbias['b1']:
            filexml.NewChild(bias_b1, 0, "Layer_1_"+str(i), str('%25.20f' % (v)));
            i+=1
            
        i=0;
        for v in self.finalbias['out']:
            filexml.NewChild(bias_out, 0, "Layer_1_"+str(i), str('%25.20f' % (v)));
            i+=1
            
        i=0;
        for v in self.finalweights['h1']:
            thisparam=filexml.NewChild(weight_h1, 0, "Param_"+str(i));
            i+=1
            j=0
            for w in v:
                filexml.NewChild(thisparam, 0, "Layer_1_"+str(j), str('%25.20f' % (w)));
                j+=1
                
        i=0;
        for v in self.finalweights['out']:
            filexml.NewChild(weight_out, 0, "Layer_1_"+str(i), str('%25.20f' % (v)));
            i+=1

        xmldoc = filexml.NewDoc();
        filexml.DocSetRootElement(xmldoc, mainnode);
        filexml.SaveDoc(xmldoc, name);
        filexml.FreeDoc(xmldoc);

