from EnergyFitterCore import EnergyFitterCore
from EnergyFitterPlotter import EnergyFitterPlotter
import pandas as pd
import numpy as np
from math import floor, ceil

class EnergyFitterSADC(EnergyFitterCore):

    def __init__(self):
        super().__init__()
        self.InputFile = "Config0.txt"
        self.Debug = False
        self.RandomSeed = 2055382905
        self.BatchSize = 1 # useless for now
        self.ActivationFunction = "relu"
        self.nNeuron = 1
        self.Minimiser = "adam"
        self.LearningRate = 0.001
        self.nIteration = 10000
        self.TrainingFraction = 0.5
        self.CrossValidationFraction = 0.25    

    def Predictor(self, x, weights, biases):
        out_layer = x * weights['out']
        out_layer = out_layer + biases['out']
        return out_layer
    
    def InitialiseFitParameter(self):
        n_input = self.x_train.shape[1]
        print ("n_input",n_input)
        n_classes = 1
        
        self.weights = {
            'out': self.tf.Variable(self.tf.random_normal([1, n_classes]))
        }
        
        self.biases = {
            'out': self.tf.Variable(self.tf.random_normal([n_classes]))
        }
        
        
        self.x = self.tf.placeholder("float", [None, n_input])
        self.y = self.tf.placeholder("float", [None, n_classes])

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
        train_x['SumADC'] = np.float32(self.InputData.SumADC)
        for key in train_x:
            self.Feature.append(key)

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
        
        self.x_mean   = np.float32([np.mean(self.x_train)])
        self.x_stddev = np.float32([np.std (self.x_train)])
        self.y_mean   = np.mean(self.y_train)
        self.y_stddev = np.std (self.y_train)

        if self.DoNormalisation:
            print("Renormalising the data")
            print(self.x_train.shape)
            self.x_train = self.TransformToNormalised(self.x_train, self.x_mean, self.x_stddev)
            self.x_cross = self.TransformToNormalised(self.x_cross, self.x_mean, self.x_stddev)
            self.x_test  = self.TransformToNormalised(self.x_test , self.x_mean, self.x_stddev)
            self.y_train = self.TransformToNormalised(self.y_train, self.y_mean, self.y_stddev)
            self.y_cross = self.TransformToNormalised(self.y_cross, self.y_mean, self.y_stddev)
            self.y_test  = self.TransformToNormalised(self.y_test , self.y_mean, self.y_stddev)
            
        print("x_cross.shape",self.x_cross.shape)
        print("y_cross.shape",self.y_cross.shape)
        print("x_train.shape",self.x_train.shape)
        print("y_train.shape",self.y_train.shape)
        print("x_test .shape",self.x_test .shape)
        print("y_test .shape",self.y_test .shape)
            
        print("Finished instantiating the data")
        self.DoneInputDataInitialisation=True

def main():
    EFADC = EnergyFitterSADC()
    EFADC.InitialiseInputData()
    EFADC.Train()

    EFitters = [EFADC]
    EFP = EnergyFitterPlotter()
    EFP.PlotData("SADC_output.pdf", EFitters)

    EFADC.ExportXML("Simple.xml")

    
if __name__ == "__main__":
    main()


    



