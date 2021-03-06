from EnergyFitterCore import EnergyFitterCore



class EnergyFitterSigmoid(EnergyFitterCore):

    def __init__(self):
        super().__init__()
        self.InputFile = "Config2.txt"
        self.Debug = False#True
        self.RandomSeed = 2055382905
        self.BatchSize = 1 # useless for now
        self.ActivationFunction = "relu"
        self.nNeuron = 100
        self.Minimiser = "adam"
        self.LearningRate = 0.001
        self.nIteration = 100
        self.TrainingFraction = 0.5
        self.CrossValidationFraction = 0.25    


def main():
    EFS = EnergyFitterSigmoid()
    EFS.InitialiseInputData()
    EFS.Train()
    EFS.ExportXML("file_config2.xml")
    
    
if __name__ == "__main__":
    main()


    



