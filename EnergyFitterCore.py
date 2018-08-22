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
        self.DoNormalisationInput = True
        self.DoNormalisationOutput = True
        self.DisplayStep=100
        self.InputFile = ""
        self.InputData = pd.DataFrame()

        self.Name = ""

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
        
        self.Feature = []
        self.Output  = []
        self.nFeature = 0
        self.nOutput  = 0
        
        self.x_cross = np.zeros(1)
        self.y_cross = np.zeros(1)
        self.x_train = np.zeros(1)
        self.y_train = np.zeros(1)
        self.x_test  = np.zeros(1)
        self.y_test  = np.zeros(1)
        self.FinalPredictionCrossVal = np.zeros(1)

        self.tf = tf
        self.x = self.tf.placeholder("float", [None, None])
        self.y = self.tf.placeholder("float", [None, None])
        
        self.x_mean   = np.zeros(1)
        self.x_stddev = np.ones(1)
        self.y_mean   = 0
        self.y_stddev = 1
        self.weights = {}
        self.bias    = {}

        self.finalweights = {}
        self.finalbias    = {}

    def Execute(self):
        self.InitialiseInputData()
        self.Train()

  
    def TransformToNormalised(self, x, mean, stddev):
        x_normalised = (x-mean)/stddev
        return x_normalised

    def TransformToReal(self, x, mean, rms):
        return x*rms+mean

    def Predictor(self, x, weights, biases):
        layer_1 = self.tf.matmul(x, weights['h1'])
        layer_1 = self.tf.add(layer_1, biases['b1'])
        layer_1 = self.tf.nn.relu(layer_1)
        out_layer = self.tf.matmul(layer_1, weights['out'])
        out_layer = out_layer + biases['out']
        return out_layer
    #return out_layer
    
    def InitialiseInputData(self):
        if self.InputFile=="":
            print("Input file wasn't set, do EnergyFitterCore.InputFile=\"File.csv\"")
            return

        print("Input file is "+self.InputFile)
        # Read the csv
        self.InputData = pd.read_csv(self.InputFile, sep=",")
        # Shuffle it
        self.InputData = self.InputData.sample(frac=1)
        if self.Debug:
            print(self.InputData)

        self.InputData = self.InputData[self.InputData['Type']==1]

        if self.Debug:
            print(self.InputData)
        # Define 2 pandas which contains the training data
        train_x = pd.DataFrame()
        for feat in self.Feature:
            print(feat)
            if feat in self.InputData.columns:
                train_x[feat] = np.float32(self.InputData[feat])
            else:
                print("The feature "+feat+" doesn't exist in the input data (check "+self.Name+" section in the config file).")
                exit()
            
        train_x = pd.concat([train_x], axis=1)

        train_y = pd.DataFrame()
        for out in self.Output:
            if feat in self.InputData.columns:
                train_y[out] = np.float32(self.InputData[out])
            else:
                print("The feature "+feat+" doesn't exist in the input data (check "+self.Name+" section in the config file).")
                exit()
                

        self.nFeature = train_x.shape[1]
        self.nOutput  = train_y.shape[1]
 
        cross_cnt = floor(train_x.shape[0] * self.CrossValidationFraction)
        train_cnt = floor(train_x.shape[0] * (self.TrainingFraction+
                                              self.CrossValidationFraction))
                
        self.x_cross = train_x.iloc[0:cross_cnt].values
        self.y_cross = train_y.iloc[0:cross_cnt].values
        self.x_train = train_x.iloc[cross_cnt:train_cnt].values
        self.y_train = train_y.iloc[cross_cnt:train_cnt].values
        self.x_test  = train_x.iloc[train_cnt:].values
        self.y_test  = train_y.iloc[train_cnt:].values
        if self.Debug:
            print("self.x_train ",      self.x_train)
            print("self.y_train ",      self.y_train)
        print(self.nFeature)
        print(type([np.mean(self.x_train[:,i]) for i in range(0,self.nFeature)]))

        self.x_mean   = np.float32([np.mean(self.x_train[:,i]) for i in range(0,self.nFeature)])
        self.x_stddev = np.float32([np.std (self.x_train[:,i]) for i in range(0,self.nFeature)])
        self.y_mean   = np.float32([np.mean(self.y_train[:,i]) for i in range(0,self.nOutput)])
        self.y_stddev = np.float32([np.std (self.y_train[:,i]) for i in range(0,self.nOutput)])
        if self.Debug:
            print("self.x_mean   ", self.x_mean  )
            print("self.x_stddev ", self.x_stddev)
            print("self.y_mean   ", self.y_mean  )
            print("self.y_stddev ", self.y_stddev)
        
        if self.DoNormalisationInput:
            print("Renormalising the input data")
            self.x_train = [self.TransformToNormalised(self.x_train[:,i], self.x_mean[i], self.x_stddev[i]) for i in range(0,self.nFeature)]
            self.x_cross = [self.TransformToNormalised(self.x_cross[:,i], self.x_mean[i], self.x_stddev[i]) for i in range(0,self.nFeature)]
            self.x_test  = [self.TransformToNormalised(self.x_test [:,i], self.x_mean[i], self.x_stddev[i]) for i in range(0,self.nFeature)]
            self.x_train = np.transpose(self.x_train)
            self.x_cross = np.transpose(self.x_cross)
            self.x_test  = np.transpose(self.x_test )

        if self.DoNormalisationOutput:
            print("Renormalising the output data")
            self.y_train = [self.TransformToNormalised(self.y_train[:,i], self.y_mean[i], self.y_stddev[i]) for i in range(0,self.nOutput)]
            self.y_cross = [self.TransformToNormalised(self.y_cross[:,i], self.y_mean[i], self.y_stddev[i]) for i in range(0,self.nOutput)]
            self.y_test  = [self.TransformToNormalised(self.y_test [:,i], self.y_mean[i], self.y_stddev[i]) for i in range(0,self.nOutput)]
            self.y_train = np.transpose(self.y_train)
            self.y_cross = np.transpose(self.y_cross)
            self.y_test  = np.transpose(self.y_test )
            
            

        print("x_cross.shape",self.x_cross.shape)
        print("y_cross.shape",self.y_cross.shape)
        print("x_train.shape",self.x_train.shape)
        print("y_train.shape",self.y_train.shape)
        print("x_test .shape",self.x_test .shape)
        print("y_test .shape",self.y_test .shape)
            
        print("Finished instantiating the data")
        self.DoneInputDataInitialisation=True



    def InitialiseFitParameter(self):
        self.weights = {
            'h1':  self.tf.Variable(self.tf.random_normal([self.nFeature, self.nNeuron])),
            'out': self.tf.Variable(self.tf.random_normal([self.nNeuron,  self.nOutput]))
        }
        
        self.biases = {
            'b1':  self.tf.Variable(self.tf.random_normal([self.nNeuron])),
            'out': self.tf.Variable(self.tf.random_normal([self.nOutput]))
        }
                
        self.x = self.tf.placeholder("float", [None, self.nFeature])
        self.y = self.tf.placeholder("float", [None, self.nOutput])
        




        
    def Train(self):
        self.InitialiseFitParameter()
        predictions = self.Predictor(self.x, self.weights, self.biases)
        cost = self.tf.reduce_mean(self.tf.losses.mean_squared_error(predictions=predictions, labels=self.y))
        optimizer = self.tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(cost)

        with self.tf.Session() as sess:
            sess.run(self.tf.global_variables_initializer())
            
            print("Optimization for" +self.Name+ " neurons starting!")

            for epoch in range(self.nIteration):
                x_batch = self.x_train
                y_batch = self.y_train
                if self.Debug:
                    if epoch>=1:
                        break
                    print("EPOCH",epoch)
                    print("x_batch ",      x_batch)
                    print("y_batch ",      y_batch)
                    print("biases ",       sess.run(self.biases['b1']))
                    print("outbias ",      sess.run(self.biases['out']))
                    print("weights[h1] ",  sess.run(self.weights['h1']))
                    print("weights[out] ", sess.run(self.weights['out']))


                _, c_train, pred = sess.run([optimizer, cost, predictions], feed_dict={self.x: x_batch, self.y: y_batch})
                if self.Debug:
                    print("ctrain ", c_train)
                    print("pred ",   pred)
                    print("len ctrain ", c_train.shape)
                    print("len pred ",   pred.shape)

                c_cross, _ = sess.run([cost, predictions], feed_dict={self.x: self.x_cross, self.y: self.y_cross})
                
                
                self.CrossValidationCost.append(c_cross)
                self.TrainCost          .append(c_train)
                avg_cost = c_train
                
                if epoch % self.DisplayStep == 0 or self.Debug:
                    print("Epoch:", '%06d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

            self.finalweights = sess.run(self.weights) 
            self.finalbias    = sess.run(self.biases )
            print("Optimization for" +self.Name+ " neurons finished!")
            self.FinalPredictionCrossVal = sess.run(predictions, feed_dict={self.x: self.x_cross})
            self.DoneMinimisation=True


    def Validate(self,x_testing):
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







            
    def ExportXML(self, name):
        if not self.DoneMinimisation:
            print("The minimisation wasn't done, not saving the XML")
            return
        filexml = xml()
        mainnode = filexml.NewChild(0, 0, "main");
        param = filexml.NewChild(mainnode, 0, "Param")

        filexml.NewChild(param, 0, "nNeuron",    str(self.nNeuron));    
        filexml.NewChild(param, 0, "nParam",     str(len(self.Feature)));    
        filexml.NewChild(param, 0, "nOutput",    str(1));    
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
        
        bias   = filexml.NewChild(mainnode, 0, "Bias"  );
        weight = filexml.NewChild(mainnode, 0, "Weight");
        print(type(self.finalbias))
        for key, values in self.finalbias.items():
            bias_lay = filexml.NewChild(bias, 0, key);
            i=0;
            for val in values:
                filexml.NewChild(bias_lay, 0, "Layer_1_"+str(i), str('%25.20f' % (val)));
                i+=1
            
        for key, values in self.finalweights.items():
            weight_lay = filexml.NewChild(weight, 0, key)
            i=0;
            for param in values:
                weight_par = filexml.NewChild(weight_lay, 0, "Param_"+str(i));
                i+=1
                j=0
                for val in param:
                    filexml.NewChild(weight_par, 0, "Layer_1_"+str(j), str('%25.20f' % (val)));
                    j+=1
                
        xmldoc = filexml.NewDoc();
        filexml.DocSetRootElement(xmldoc, mainnode);
        filexml.SaveDoc(xmldoc, name);
        filexml.FreeDoc(xmldoc);

        
