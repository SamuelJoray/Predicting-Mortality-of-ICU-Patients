import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import CIT
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.HiddenCausal.GIN.GIN import GIN

import networkx as nx
import copy

class CausalGraph:
    
    def __init__(self,data,time_range=8):
        self.data = data
        self.graph=None
        self.variable_names = self.data.columns
        self.nb_variables= len(self.variable_names)
        self.time_range=time_range
        self.predictors = []
        self.coefficients = []
        self.intercept = 0
        self.intercept0 = 0
        self.root_nodes_values = []
        self.coef_matrix=[]
        self.root_node_means=[]
        self.bk = BackgroundKnowledge()
        self.adjacency_matrix=[]
    

    #the ordering of the columns in the dataset is assumed to be first the variables that depends on time, 
    #then the 4 variables that does not depend on time in the following order: Age, Weight, Gender, Height
    #and the last one is the variable outcomes.
    def setBackgroundknowledge(self, time_variables_range = []):  
        
        if len(time_variables_range) == 0:
            time_variables_range = [0,self.nb_variables-5]
            
        nodes=[]
        for node_name in self.variable_names:
            nodes.append(GraphNode(node_name))
    
        
        for node1 in range(time_variables_range[0], time_variables_range[1], self.time_range):#48
            for node2 in range(time_variables_range[0], time_variables_range[1], self.time_range):
                for i in range(0, self.time_range):
                    for j in range(0, self.time_range):
                        if j<i or j>i+1:
                            self.bk.add_forbidden_by_node(nodes[node1+i], nodes[node2+j])
    
        for node in range(self.nb_variables-1):
            self.bk.add_forbidden_by_node(nodes[self.nb_variables-1],nodes[node] )#only incoming arrows in outcome
            
        for node in range(self.nb_variables-1):    
            self.bk.add_forbidden_by_node(nodes[node],nodes[self.nb_variables-5] )#no incoming arrows in age

        for node in range(self.nb_variables-1):
            self.bk.add_forbidden_by_node(nodes[node],nodes[self.nb_variables-3] )#no incoming arrows in gender
                
        for node in range(self.nb_variables-1):
            self.bk.add_forbidden_by_node(nodes[node],nodes[self.nb_variables-2] )#no incoming arrows in height
                
        for node in range(self.nb_variables-2):
            self.bk.add_forbidden_by_node(nodes[node],nodes[self.nb_variables-4] )#no incoming arrows in weight
            
        
        
    def findGraph(self):
        self.graph = pc(self.data.values,indep_test="fisherz",  alpha=0.05,background_knowledge=self.bk,node_names=self.variable_names)
        
    
    
        
    def drawSaveGraph(self,directory):
        self.graph.draw_pydot_graph(self.variable_names)
        GraphUtils.to_pydot(self.graph.G,labels=self.variable_names).write_png(directory+'causal_graph.png')
        
    def saveGraph(self,directory):
        with open(directory+'causal_graph.pkl', 'wb') as file:
            pickle.dump(self.graph, file)
        
    def openGraph(self,directory):
        with open(directory+'causal_graph.pkl', 'rb') as file:
            self.graph = pickle.load(file)

    
    def getParents(self,child):
        parents=[]
        graph_nodes_name = [node.get_name() for node in self.graph.G.nodes]
        index_node = graph_nodes_name.index(child)
        for i,variable_name in enumerate(graph_nodes_name):
            if self.graph.G.graph[index_node,i]==1 and self.graph.G.graph[i,index_node]==-1:
                parents.append(variable_name)
        return parents
    
    
    
    def getChilds(self,parent):
        childs=[]
        graph_nodes_name = [node.get_name() for node in self.graph.G.nodes]
        index_node = graph_nodes_name.index(parent)
        for i,variable_name in enumerate(graph_nodes_name):
            if self.graph.G.graph[index_node,i]==-1 and self.graph.G.graph[i,index_node]==1:
                childs.append(variable_name)
        return childs
        
   
    
    def hasParents(self,child):
        graph_nodes_name = [node.get_name() for node in self.graph.G.nodes]
        index_node = graph_nodes_name.index(child)
        if 1 in self.graph.G.graph[index_node,:]:
            return True
        else:
            return False
  
  
    
        
        

    def setCoefMatrix(self):

        coef_matrix = np.zeros((self.nb_variables,self.nb_variables+1))
        coef_importance = np.zeros((self.nb_variables,self.nb_variables))
        for i in range(self.nb_variables):
            pred=[]
            for j in range(self.nb_variables):
                if self.graph.G.graph[j,i] == -1 and self.graph.G.graph[i,j]==1 and i!=j:
                    pred.append(j)
             
                    
            if pred != []:
                #X = data[:,pred]
                #y = data[:,i]
                index1 = self.data[self.variable_names[pred]].dropna().index
                index2 = self.data[self.variable_names[i]].dropna().index
                index =  np.intersect1d(index1, index2)
    
                X = self.data[self.variable_names[pred]].iloc[index].values
                y =  self.data[self.variable_names[i]].iloc[index].values
                
                model = LinearRegression()
                
                try: 
                    model.fit(X, y)
                except Exception as e:
                   pass
                    
                
                coef_matrix[i,np.array(pred)+1] = model.coef_
                coef_matrix[i,0] = model.intercept_
                
                
                
                feature_stds = np.std(X, axis=0)
                
                # Calculate the standard deviation of the target variable
                target_std = np.std(y)
                
                # Calculate standardized coefficients
                standardized_coefficients = model.coef_ * (feature_stds / target_std)
                
                coef_importance[i,np.array(pred)] = standardized_coefficients
            
        self.coef_matrix = pd.DataFrame(coef_matrix,columns=np.concatenate((["intercept"],self.variable_names)),index = self.variable_names )
        self.coef_importance = pd.DataFrame(coef_importance,columns=self.variable_names,index = self.variable_names )
    
    
    def setAdjacencyMatrix(self):
        
        matrix_graph=copy.deepcopy(self.graph.G.graph)
        self.adjacency_matrix=matrix_graph
        for i in range(np.shape(matrix_graph)[0]):
            for j in range(np.shape(matrix_graph)[1]):
                if matrix_graph[i,j] == -1 and matrix_graph[j,i]==-1:
                    self.adjacency_matrix[i,j]=0
                    self.adjacency_matrix[j,i]=0
                
        for i in range(np.shape(matrix_graph)[0]):
            for j in range(np.shape(matrix_graph)[1]):
                if matrix_graph[i,j]==-1:
                    self.adjacency_matrix[i,j]=0

    
    
        
    def getCycle(self):
        
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        
        # Check for cycles
        try:
            cycle = nx.find_cycle(G, orientation="original")
            return cycle
        except nx.exception.NetworkXNoCycle:
           return []
        
    
    def removeCycle(self,cycle):
        
        matrix_graph=self.graph.G.graph
        
        if cycle !=[]:
            
            if matrix_graph[cycle[0][0],cycle[0][1]]== 1:
                matrix_graph[cycle[0][0],cycle[0][1]]=-1
            elif matrix_graph[cycle[0][1],cycle[0][0]]== 1:
                matrix_graph[cycle[0][1],cycle[0][0]]=-1
                
        self.graph.G.graph = matrix_graph
    
    
    
          
    def setRootNodesMeans(self):
        root_node_names =[]
        root_node_means = []
        # Iterate over all nodes in the graph
        for variable_name in self.variable_names:
            # Check if the node has no parents (no incoming edges)
            if not self.hasParents(variable_name):
                root_node_names.append(variable_name)
                root_node_means.append(np.mean(self.data[variable_name].dropna()))
        
        root_node_means_dataframe=pd.DataFrame(np.array(root_node_means).reshape(1,-1),columns=root_node_names)
        self.root_node_means = root_node_means_dataframe
    
    
    
    
    def setPredictors(self,variable_name_individual,node,predictors,coefficients,new_coeff,new_intercept,root_nodes_values):
        for parent in self.getParents(node):
            if (not parent in variable_name_individual) and self.hasParents(parent):
                new_intercept = self.setPredictors(variable_name_individual,parent,predictors,
                                               coefficients,new_coeff*self.coef_matrix[parent][node],
                                               new_intercept+new_coeff*self.coef_matrix[parent][node]*self.coef_matrix["intercept"][parent],root_nodes_values)[2]
            elif not parent in predictors:
                if parent in variable_name_individual:
                    
                    
                    coefficients.append(new_coeff*self.coef_matrix[parent][node])
                    predictors.append(parent)
                else:
                   
                    root_nodes_values.append(new_coeff*self.coef_matrix[parent][node]*self.root_node_means[parent])
                    
              
        return predictors, coefficients, new_intercept, self.coef_matrix["intercept"][node], root_nodes_values
         
    
    
    def getPrediction(self, data_test):#data_test must be for one individual
     
        prediction = np.dot(data_test[self.predictors].values, self.coefficients) + self.intercept + self.intercept0+np.sum(self.root_nodes_values)
        
        return prediction
    
    
    
   
        
        
        
        
        
        
        
        
        