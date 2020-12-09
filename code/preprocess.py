# Import dependencies

import numpy as np
import math
from numpy import sort
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
sns.set(style='white', palette = 'Paired')
#plt.style.use('ggplot')
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
np.set_printoptions(suppress=True) # Suppress scientific notation where possible
from ipywidgets import interactive, FloatSlider

class preprocess():
    def __init__(self, train):
        self.train = train
        
    def clean(self, df):
        """
        Cleans the data
        """
        # Remove id column
        df.drop(columns='id', inplace=True)
        
        # Remove categorical columns, as they appear to contain meaningless information
        df = df[list(df._get_numeric_data().columns)]
        
        # Remove duplicate columns
        df = df.T.drop_duplicates().T
        
        # Convert to float
        df = df.astype('float64')
        
        # Fill NaN values
        df.fillna(df.mean(), inplace=True)
        
        return df
    
    def missing_values_table(self, df):
        """
        Takes the dataframe and returns missing values and total percentage of values
        """
        
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    def transform_collinear_features(self, df, threshold):
        '''
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model
            to generalize and improves the interpretability of the model.

        Inputs: 
            threshold: any features with correlations greater than this value are removed

        Output: 
            dataframe that replaces correlated columns with their differences
        '''

        # Dont want to remove correlations between loss
        y = df['loss']
        x = df.drop(columns = ['loss'])

        # Calculate the correlation matrix
        corr_matrix = df.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []
        drop_rows= []
        col_list1 = []
        col_list2 = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the thresholds
                if val >= threshold:
                    # Print the correlated features and the correlation value
                    #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))

                    drop_cols.append(col.values[0])
                    drop_rows.append(row.values[0])

                    # Put columns in separate lists
                    col_list1.append(col.values[0])
                    col_list2.append(row.values[0])


        # zip the 2 lists together
        col_list = list(zip(col_list1, col_list2))

        # create new dataframe to store transformed columns
        trans_cols = pd.DataFrame()

        # create new columns in new dataframe, keeping the differences in the correlated columns
        for i in col_list:
            trans_cols[i] = df[i[0]] - df[i[1]]

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        df = df.drop(columns = drops)
        drops = set(drop_rows)
        df = df.drop(columns=drops, errors='ignore')

        # Add the score back in to the data
        df['loss'] = y
        trans_cols['loss'] = y
        
        # Combine dataframes
        df = pd.concat([df.drop(columns='loss'), trans_cols], axis=1)

        return df
    
    def make_classes(self, df):
        """
        Creates target classes of 0 for all with 0 loan loss and 1 for all loan losses of >0
        """
        
        for i in range(len(df['loss'])):
            if df['loss'][i] > 0:
                df['loss'][i] = 1
                
        return df