from EnergyFitterCore import EnergyFitterCore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


material_palette = ["#4CAF50", "#2196F3", "#9E9E9E", "#FF9800", "#607D8B", "#9C27B0"]

class EnergyFitterPlotter:
    
    def PlotData(self, outputfile, EFCs):
        pp = PdfPages(outputfile)

        TrainingOutput = {}
        bins = np.linspace(0, 50, 51)
        cost_after_min = {}
        for EFC in EFCs:
            cost_after_min[EFC.nNeuron] = EFC.CrossValidationCost[-1]
            fig, ax = plt.subplots(figsize=(20, 12))
            fig.suptitle("Cost")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost")
            print(len(EFC.CrossValidationCost))
            trainplot = plt.semilogy(range(len(EFC.CrossValidationCost)), EFC.CrossValidationCost, 'o-')
            crossplot = plt.semilogy(range(len(EFC.TrainCost          )), EFC.TrainCost          , 'o-')
            plt.legend([trainplot[0], crossplot[0]], ('Training set', 'Cross validation set'))
            plt.grid(True)
            fig.savefig(pp, format='pdf')
            
            this_output = {}
            this_output['predicted'] = np.float32(EFC.TransformToReal(EFC.FinalPredictionCrossVal[:,0], EFC.y_mean, EFC.y_stddev))
            this_output['true']      = np.float32(EFC.TransformToReal(EFC.y_cross   [:,0], EFC.y_mean, EFC.y_stddev))
            fig, ax = plt.subplots(figsize=(20, 12))
            spectrum_pred = plt.hist(x=this_output['predicted'], bins=10, range=[0,50], label='EReco', lw=3,fc=(1, 0, 0, 0),ec=(1, 0, 0, 1))
            spectrum_true = plt.hist(x=this_output['true'],      bins=10, range=[0,50], label='ETrue', lw=3,fc=(0, 0, 1, 0),ec=(0, 0, 1, 1))
            plt.legend()
            ax.set_xlabel('E [MeV]')
            plt.grid(True)
            fig.savefig(pp, format='pdf')
            
            
            digitized = np.digitize(this_output['true'], bins)
            this_output['true_binned'] = np.float32(digitized)
            this_output['bias'] = np.float32(this_output['predicted'] - this_output['true'])
            this_output['rms']  = np.float32((this_output['predicted'] - this_output['true']) * (this_output['predicted'] - this_output['true']))

            TrainingOutput[EFC] = this_output

            for i in range(0,this_output['true'].shape[0]):
                if this_output['predicted'][i]>100 or this_output['predicted'][i]<0:
                    print("enu  ", this_output['predicted'][i])
                    print("erec ", this_output['true'][i])
                    
            fig, ax = plt.subplots(figsize=(20, 12))
            fig.suptitle("Smearing")
            ax.set_xlabel('E True [MeV]')
            ax.set_ylabel('E Reco [MeV]')
            plt.hist2d(this_output['true'],this_output['predicted'], bins=100, range=[[0,100],[0,100]],norm=colors.LogNorm())
            plt.colorbar()
            plt.grid(True)
            fig.savefig(pp, format='pdf')

            fig, ax = plt.subplots(figsize=(20, 12))
            fig.suptitle("Smearing")
            ax.set_xlabel('E True [MeV]')
            ax.set_ylabel('E Reco [MeV]')
            sns.regplot(x=this_output['true_binned'],y=this_output['predicted'],x_bins=bins, fit_reg=None)
            plt.grid(True)


        fig, (axs1,axs2) = plt.subplots(2, 1, sharex='col',figsize=(20, 12))
        fig.subplots_adjust(hspace=0)
        fig.suptitle("Resolution and Bias")
        plt.subplot(2, 1, 1)
        i=0
        for EFitter, Output in TrainingOutput.items():
            lab = '{0} neurons, {1}, '.format(EFitter.nNeuron,EFitter.ActivationFunction)
            for f in EFitter.Feature:
                lab+= f+" "
            sns.regplot(x=Output['true_binned'],y=Output['rms'], x_bins=bins, fit_reg=None,color=material_palette[i],
                        label=lab)
            i+=1
        plt.grid(True)
        plt.ylabel('Relative Error [MeV]')
        plt.legend()
        plt.subplot(2, 1, 2)
        i=0
        for EFitter, Output in TrainingOutput.items():
            sns.regplot(x=Output['true_binned'],y=Output['bias'], x_bins=bins, fit_reg=None,color=material_palette[i])
            i+=1
        plt.grid(True)
        plt.xlabel('E True [MeV]')
        plt.ylabel('Bias (Reco-True) [MeV]')
        fig.savefig(pp, format='pdf')
        
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.suptitle("Cost")
        ax.set_xlabel("Number of hidden layers")
        ax.set_ylabel("Cost")
        keys = np.fromiter(cost_after_min.keys(),   dtype=float)
        vals = np.fromiter(cost_after_min.values(), dtype=float)
        plt.plot(keys,vals, 'o-')
        fig.savefig(pp, format='pdf')
        pp.close()
