import argparse
import configparser as cfp
from EnergyFitterCore import EnergyFitterCore as efc
from EnergyFitterPlotter import EnergyFitterPlotter as efp





def SetupFitter(name,config):
    Fitter = efc()
    Fitter.Name = name
    Fitter.InputFile = config.get('InputFile')
    Fitter.RandomSeed = int(config.get('RandomSeed'))
    Fitter.DoNormalisationInput  = config.getboolean('NormaliseInput')
    Fitter.DoNormalisationOutput = config.getboolean('NormaliseOutput')
    Fitter.Feature = config.get('InputParam').split(',')
    print(Fitter.Feature)
    Fitter.Output = config.get('OutputParam').split(',')
    Fitter.TrainingFraction = config.getfloat('TrainingFraction')
    Fitter.CrossValidationFraction = config.getfloat('CrossValidationFraction')
    Fitter.LearningRate = config.getfloat('LearningRate')
    Fitter.nIteration = config.getint('NIteration')
    Fitter.nNeuron = config.getint('NNeuron')
    Fitter.Debug = config.getboolean('Debug')
    return Fitter


    

def main(config):
    Fitters = []
    defopt = config['DEFAULT']

    for c in config:
        if c!="DEFAULT" and c!="PLOT" and c!="RUNTIME":
            print("Setting up fitter "+c)
            Fitters.append(SetupFitter(c,config[c]))

    for f in Fitters:
        if config['RUNTIME'].getboolean('Retrain'):
            f.Execute()
        f.ExportXML(f.Name+"_"+config['RUNTIME'].get('OutputXMLPostfix'))
        
    plotter = efp()
    if config['RUNTIME'].getboolean('SaveOutputIndividualPlot'):
        for f in Fitters:
            fts = [f]
            print(f.Name+"_"+config['RUNTIME'].get('OutputPDFPostfix'))
            filename = str(f.Name+"_"+config['RUNTIME'].get('OutputPDFPostfix'))
            plotter.PlotData(filename,fts)

    fitter_dict = {}
    for f in Fitters:
        fitter_dict[f.Name] = f

    
    for sets in config['PLOT']:
        if sets not in defopt:
            print("sets",sets)
            names=config['PLOT'][sets].split(',')
            print("names",names)
            fts = []
            for name in names:
                print("name",name)
                fts.append(fitter_dict[name])
            plotter.PlotData(sets+"_"+config['RUNTIME'].get('OutputPDFPostfix'),fts)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs the energy fitter')
    parser.add_argument('-c', help='The config file which has to be run')
    args = parser.parse_args()
    arg = vars(args)
    if arg['c'] == None:
        print("You need to pass a config file!")
        exit
        
    config = cfp.ConfigParser()
    config.read(arg['c'])
    main(config)
    
