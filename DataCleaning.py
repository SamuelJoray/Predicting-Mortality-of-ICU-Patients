import os
import numpy as np
import pandas as pd
import pickle

from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

class DataCleaning:
    
    def __init__(self,time_interval=8):
        self.data=[]
        self.df=[]
        self.variable_names=[]
        self.file_names=[]
        self.outcomes=[]
        self.dataset = pd.DataFrame()
        self.time_interval=time_interval
        self.time_range=int(48/self.time_interval)
        self.dataset_imputed= pd.DataFrame()
        
    def loadFiles(self,directory):

        # List all files in the directory
        self.file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for file_name in self.file_names:
            
            self.df.append(pd.read_csv(directory+str(file_name)))
            
            
    def loadOutcomes(self,directory):
        
        self.outcomes = pd.read_csv(directory)


    def rowsToRemove(self,rows_to_remove):


        # Remove the row with the specified RecordID
        for row_to_remove in rows_to_remove:
            self.outcomes = self.outcomes[self.outcomes['RecordID'] != row_to_remove]
            
        self.outcomes = self.outcomes.reset_index(drop=True)
        
        
    def removeDuplicates(self):
        # Process each file
        for i in range(len(self.df)):
            
            #store all the numbers attributed to the patients
            #file_names.append(str(file_name))
            
            #remove all rows where Urine=0
            ind_urine = self.df[i]["Parameter"]=="Urine" 
            ind_urine_0 = self.df[i][ind_urine]["Value"]==0
            lines_to_remove = ind_urine_0[ind_urine_0].index
            
            for j in lines_to_remove:
                
                self.df[i] = self.df[i].drop(index=j) #remove the ith row
            
            #take the mean of the duplicates that are left
            self.df[i] = self.df[i].groupby(['Time','Parameter'], as_index=False).agg({'Value': 'mean'})
            
            
        
    def transformLongToWide(self):
        for i in range(len(self.df)):
            #transform the dataset from a long to a wide format dataset and make all txt files into one dictionary data_dict
            self.df[i] = self.df[i].pivot(index='Time', columns='Parameter', values='Value').reset_index()
            self.variable_names.append(self.df[i].columns.values)

        
        self.variable_names = [item for sublist in self.variable_names for item in sublist]
        self.variable_names = set(self.variable_names)
        self.variable_names.remove("Time")
        self.variable_names=np.array(list(self.variable_names))
    
    def aggregate(self):#time_interval are integers
        
       
        
        for i in range(len(self.df)):
          
                
            self.df[i]['Time'] = pd.to_timedelta(self.df[i]['Time'] + ':00')
            
            # Set the 'Time' column as the index
            self.df[i].set_index('Time', inplace=True)
            
            
            # Resample the data to 6-hour intervals and compute the mean
            self.df[i] = self.df[i].resample(str(self.time_interval)+'H').mean()
            
            
    def imputeMissingValues(self,method="forwardFill"):#method is either "knn" or "forwardFill". Default is "forwardFill".
        
        if method == "knn":
            #handle misssing values
            #knn imputing
            #data_agg_knn = copy.deepcopy(data_agg)
            for i in range(len(self.df)):
                imputer = KNNImputer(n_neighbors=5)
                imputed_data = imputer.fit_transform(self.df[i])
                self.df[i] = pd.DataFrame(imputed_data, columns=self.df[i].columns.values)
                
    
        elif method == "forwardFill":
    
            
            #forward fill
            for i in range(len(self.df)):
                
                #forward fill
                imputed_data = self.df[i].fillna(method='ffill')
                #backward fill for the missing values remaining, i.e., the columns that starts with missing values.
                imputed_data = imputed_data.fillna(method='bfill')
                self.df[i] = pd.DataFrame(imputed_data, columns=self.variable_names)
                
        
        
    def removeOutliers(self):#remove and then impute outliers
    
            
        # Fit Isolation Forest
        for variable_name in self.variable_names:
            data=[] #we create a time serie data that contains all observations of all patients for all time but for one variable (pH,Temp,...)
            file_name_used=[]
            for i in range(len(self.df)):
                try: 
                    if not np.isnan(self.df[i][variable_name].values).any():
                        if len(data)==0:
                            #we might get an error since not all patients(files) have all variables.
                            #Therefore, we use try, except.
    
                            data=self.df[i][variable_name].values
                        else:
                            
                            
                            data = np.concatenate((data,self.df[i][variable_name].values))
                            
                    
                
                except Exception as e:
                    pass
           
        
            if len(data) !=0:
                iso_forest = IsolationForest(contamination=0.0002)
                outlier_predictions = iso_forest.fit_predict(np.array(data).reshape(-1,1)) 
                
                
                c=0
                for i in range(len(self.df)):
                    try:
                        for j in range(len(self.df[i][variable_name])):
                            if outlier_predictions[c] == -1:
                                self.df[i][variable_name].iloc[j]=np.nan
                                
                            c = c+1
                    except Exception as e:
                        pass
                    
                self.imputeMissingValues()
            
            
    def mergeDataset(self,variables_to_remove):
        
        variables = ["Age","Weight","Gender","Height"]
        var_names=[]
        for variable_name in self.variable_names:
            if (not variable_name in variables) and (not variable_name in variables_to_remove):
                for time in range(0,int(self.time_range)):
                    array=[]
                    for i in range(len(self.df)):
                        try:
                            array.append(self.df[i][variable_name][time])
                        except Exception as e:
                            array.append(np.nan)
                        
                    self.dataset[variable_name+str(time)]=array
                    var_names.append(variable_name+str(time))
                
        

        for variable in variables:
            array=[]
            c=0
            for i in range(len(self.df)):
                if not -1 in self.df[i][variable].values:
                    array.append(np.mean(self.df[i][variable].values))
                else:
                    c=c+1
                    array.append(np.nan)
            print(c)
            self.dataset[variable] = array
            var_names.append(variable)
                
        self.dataset["outcomes"]= self.outcomes["In-hospital_death"]
        var_names.append("outcomes")
        
        self.variable_names = var_names
        
        imputer = KNNImputer(n_neighbors=5)
        self.dataset_imputed = pd.DataFrame(imputer.fit_transform(self.dataset),columns=self.variable_names)


        

    def cleanDataset(self,directory,directory_outcome,rows_to_remove=[],variables_to_remove=[],method="forwardFill"):
                    
                
        self.loadFiles(directory)
        
        self.loadOutcomes(directory_outcome)
        
        self.rowsToRemove(rows_to_remove)
            
        self.removeDuplicates()
            
        self.transformLongToWide()
            
        self.aggregate()
        
        self.imputeMissingValues(method = method)
        
        self.removeOutliers()
        
        self.mergeDataset(variables_to_remove)
        
        
        

            
    def saveIntoFiles(self,directory):
                
        with open(directory+'dataset'+str(self.time_interval)+'H.pkl', 'wb') as file:
            pickle.dump(self.dataset, file)
            
        with open(directory+'dataset_imputed'+str(self.time_interval)+'H.pkl', 'wb') as file:
            pickle.dump(self.dataset_imputed, file)
            
        with open(directory+'df'+str(self.time_interval)+'H.pkl', 'wb') as file:
            pickle.dump(self.df, file)

        with open(directory+'variable_names.pkl', 'wb') as file:
            pickle.dump(self.variable_names, file)
            
        np.save(directory+'file_names.npy', self.file_names)
        
        
    def openFiles(self,directory):
                

        with open(directory+'dataset'+str(self.time_interval)+'H.pkl', 'rb') as file:
            self.dataset = pickle.load(file)
            
        with open(directory+'dataset_imputed'+str(self.time_interval)+'H.pkl', 'rb') as file:
            self.dataset_imputed = pickle.load(file)
            
        with open(directory+'df'+str(self.time_interval)+'H.pkl', 'rb') as file:
            self.df = pickle.load(file)
       # with open(directory+'data_cleaned.pkl', 'wb') as file:
        #    pickle.dump(self.df, file)
            
        
        with open(directory+'variable_names.pkl', 'rb') as file:
            self.variable_names = pickle.load(file)
            
        
        self.file_names = np.load(directory+'file_names.npy', allow_pickle=True)
                
    """

directory = 'C://Users//samjo//OneDrive//Documents//école//ETH//internship ETH//'
directory_outcome='C://Users//samjo//OneDrive//Documents//école//ETH//internship ETH//Outcomes-a.txt'
data = DataCleaning()
#data.clean_dataset(directory,directory_outcome)

data.loadFiles(directory+"set-a//")

data.loadOutcomes('C://Users//samjo//OneDrive//Documents//école//ETH//internship ETH//Outcomes-a.txt')

data.rowsToRemove([139060,140501,141264,140936])
    
data.removeDuplicates()
    
data.transformLongToWide()
    
data.aggregate(8)

data.imputeMissingValues()

data.removeOutliers()

data.saveIntoFiles(directory)

data.mergeDataset()

data.openFiles(directory)

i=0
data.df[i] = data.df[i].pivot(index='Time', columns='Parameter', values='Value').reset_index()
data.variable_names.append(data.df[i].columns.values)


i=174
imputed_data = data.df[i].fillna(method='ffill')
#backward fill for the missing values remaining, i.e., the columns that starts with missing values.
imputed_data = imputed_data.fillna(method='bfill')
data.df[i] = pd.DataFrame(imputed_data, columns=data.variable_names)


with open('C://Users//samjo//OneDrive//Documents//école//ETH//internship ETH//datasetTest.pkl', 'wb') as file:
    pickle.dump(data.df, file)
  """