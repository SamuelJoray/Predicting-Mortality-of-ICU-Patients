# -*- coding: utf-8 -*-

import sys
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
sys.path.append(directory)

import numpy as np
import pandas as pd

from DataCleaning import DataCleaning
from CausalGraph import CausalGraph
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats



class Estimator:
    
    def __init__(self, time_interval,directory):
        

        self.time_interval = time_interval
        self.data_train = pd.DataFrame()
        self.data_test = pd.DataFrame()
        self.threshold=0
        self.result=[]
        self.directory = directory
        
    def setDirectories(self):
        
        self.directory_set_a = self.directory + 'set-a//'
        self.directory_set_c = self.directory + 'set-c//'
        self.directory_outcome_a = self.directory + 'Outcomes-a.txt'
        self.directory_outcome_c = self.directory + 'Outcomes-c.txt'
        self.directory_set_a_save = self.directory + 'set_a_cleaned//'+str(self.time_interval)+'H//'
        self.directory_set_c_save = self.directory + 'set_c_cleaned//'+str(self.time_interval)+'H//'
        self.directory_graph = self.directory + 'graph//'+str(self.time_interval)+'H//'
        self.directory_plots_pred_res =  self.directory + 'plots_16H//plots_pred_res//'
        self.directory_QQ_plots =  self.directory + 'plots_16H//QQ_plots//'
        
        
    def fit(self):
        
        self.data_train = DataCleaning(time_interval=self.time_interval)
        self.data_train.cleanDataset(self.directory_set_a,self.directory_outcome_a,rows_to_remove=[139060,140501,141264,140936],variables_to_remove=["MechVent","Cholesterol","pH","RecordID"])
        self.data_train.saveIntoFiles(self.directory_set_a_save)
        
        
        self.data_test = DataCleaning(time_interval=self.time_interval)
        self.data_test.cleanDataset(self.directory_set_c,self.directory_outcome_c,variables_to_remove=["MechVent","Cholesterol","pH"])
        self.data_test.saveIntoFiles(self.directory_set_c_save)
        
        
        self.graph = CausalGraph(self.data_train.dataset_imputed,time_range=int(48/self.time_interval))
        self.graph.setBackgroundknowledge()
        self.graph.findGraph()
        self.graph.setCoefMatrix()
        self.graph.setAdjacencyMatrix()
        cycle = self.graph.getCycle()
        #while cycle != []:
        self.graph.removeCycle(cycle)
        #cycle = self.graph.getCycle()
        self.graph.drawSaveGraph(self.directory_graph)
        self.graph.saveGraph(self.directory_graph)
        #self.graph.openGraph(directory_graph)
        self.graph.setRootNodesMeans()
        
    def openFiles(self):
        self.data_train = DataCleaning(time_interval=self.time_interval)
        self.data_train.openFiles(self.directory_set_a_save)
        
        self.data_test = DataCleaning(time_interval=self.time_interval)
        self.data_test.openFiles(self.directory_set_c_save)
     
        self.graph = CausalGraph(self.data_train.dataset_imputed,time_range=int(48/self.time_interval))
        self.graph.openGraph(self.directory_graph)
        
        self.graph.setCoefMatrix()
        self.graph.setRootNodesMeans()
        
    def setThreshold(self):
        
        y_pred = self.predict(self.data_train.dataset)
        y_true = self.data_train.dataset["outcomes"]
            
        threshold_grid = np.arange(0.25,0.35,0.001)
        score=[]
        for threshold in threshold_grid:
            y_pred_custom = (np.array(y_pred) > threshold).astype(int)
            conf_matrix = confusion_matrix(y_true, y_pred_custom)
            
            sensitivity = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
            positive_predictivity = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
            
            score.append(min(sensitivity,positive_predictivity))
            
        index = np.where(score==max(score))
        self.threshold = threshold_grid[index]
    
    def predict(self,data):
        
        predictions = []
        for i in data.index:
            self.graph.predictors, self.graph.coefficients, self.graph.intercept, self.graph.intercept0,self.graph.root_nodes_values= self.graph.setPredictors(
                data.iloc[i].dropna().index.values,"outcomes",[],[],1,0,[])
            predictions.append(self.graph.getPrediction(data.iloc[i].dropna()))
        
        return predictions
    
    def getScore(self):
        
        y_pred = self.predict(self.data_test.dataset)
        y_test=self.data_test.dataset["outcomes"].values.astype(int)
        
        self.setThreshold()
        print("Threshold is : " + str(self.threshold[0]))
        y_pred_custom = (np.array(y_pred) > self.threshold[0]).astype(int)
    
        conf_matrix = confusion_matrix(y_test, y_pred_custom)
        
        sensitivity = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
        positive_predictivity = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
        
        self.result.append(min(sensitivity,positive_predictivity))
        
        print("score event 1 : " + str(min(sensitivity,positive_predictivity)))
        print("percentage of 1's : "+ str(sum(y_pred_custom)/len(y_pred_custom)))
        print("percentage of 1's true : "+ str(sum(y_test)/len(y_test)))
    
    
    def plotResiduals(self):
        
        nodes = self.graph.graph.G.nodes
        #check whether the relation between the variables are linear and have gaussian errors
        for node1 in range(len(self.graph.variable_names)):
            X=[]
            for node2 in range(len(self.graph.variable_names)):
                if self.graph.graph.G.is_directed_from_to(nodes[node2], nodes[node1]):
        
                    if X==[]: #check if it is empty
                        X = self.data_train.dataset_imputed[self.data_train.variable_names[node2]].values
                    else:
                        X = np.column_stack((X,self.data_train.dataset_imputed[self.data_train.variable_names[node2]].values))
            if len(X)!=0:
               
                
                y = self.data_train.dataset_imputed[self.data_train.variable_names[node1]]
            
               
                model = LinearRegression()
                if X.ndim==1:#check if there is only one or more features
                    model.fit(X.reshape(-1, 1), y)
                else:
                    model.fit(X, y)
            
                if X.ndim==1:
                    y_pred = model.predict(X.reshape(-1, 1))
                else:
                    y_pred = model.predict(X)
                    
                residuals = y - y_pred
                
                # Plot residuals vs predicted values
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_pred, y=residuals)
                plt.axhline(0, color='red', linestyle='--')
                plt.xlabel('Predicted Probabilities')
                plt.ylabel('Residuals')
                plt.title('Residuals vs Predicted Probabilities '+self.data_train.variable_names[node1])
                plt.savefig(self.directory_plots_pred_res + 'plot_pred_res_'+ str(self.data_train.variable_names[node1]) + '.pdf', format='pdf')
                plt.show()
                
             
                
                # Q-Q plot of residuals
                plt.figure(figsize=(10, 6))
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Q-Q Plot of Residuals '+self.data_train.variable_names[node1])
                plt.savefig(self.directory_QQ_plots + 'QQ_plot_'+ str(self.data_train.variable_names[node1]) + '.pdf', format='pdf')
                plt.show()

"""
#This clean the data, compute the causal graph and get the score for event 1. It may take some time.
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
time_intervals= [16]
for time_interval in time_intervals:
    est = Estimator(time_interval=time_interval,directory=directory)
    est.setDirectories()
    est.fit()
    
    est.getScore()
"""

"""
#From now on, it uses the data and graph that has already been saved, it will save time for computation.
#Compute the score using the data that has already been computed and has been saved
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
time_intervals = [4,8,12,16]

for time_interval in time_intervals:
    est = Estimator(time_interval=time_interval,directory=directory)
    est.setDirectories()
    
    est.openFiles()
    
    est.getScore()
    
#Compute the parents of the outcomes variable.
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
time_intervals = [4,8,12,16]  
for time_interval in time_intervals:
    est = Estimator(time_interval=time_interval,directory=directory)
    est.setDirectories()
    
    est.openFiles()
    
    print("parents of outcomes : "+str(est.graph.getParents("outcomes")))
    
   
#Plot and save the residuals and QQ plots of each variables in the graph, always using the parents of the variable as predictor.
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
time_intervals = [16]
for time_interval in time_intervals:
    est = Estimator(time_interval=time_interval,directory=directory)
    est.setDirectories()
    
    est.openFiles()
    
    est.plotResiduals()
    
    
#Compute the coefficients for each predictors of the outcomes.
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
time_intervals = [4,8,12,16]
for time_interval in time_intervals:
    est = Estimator(time_interval=time_interval,directory=directory)
    est.setDirectories()
    
    est.openFiles()
    
    parents = est.graph.getParents("outcomes")

    print(est.graph.coef_matrix[parents].loc["outcomes"])
    
    

#compute the importance of each variables.
directory = 'PATH' #'PATH' must be changed with the directory where the project is loaded.
time_intervals = [4,8,12,16]
for time_interval in time_intervals:
    est = Estimator(time_interval=time_interval,directory=directory)
    est.setDirectories()
    
    est.openFiles()
    
    parents = est.graph.getParents("outcomes")
    est.graph.setCoefMatrix()
    
    print(est.graph.coef_importance[parents].loc["outcomes"])


"""

    

