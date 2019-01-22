
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np


# In[7]:


def dropNA(df, axis = 'rows', how = 'all'):
    try:
        rows = df.shape[0]
        goals = []
        
        if axis == 'rows':
            columns = df.shape[1]
            for i in range(rows):
                if how == 'all':
                    if sum(df.iloc[i].isna()) == columns:
                        goals.append(i)
                elif how == 'any': 
                    if sum(df.iloc[i].isna()) != 0:
                        goals.append(i)
            return df.drop(goals)

        elif axis == 'columns':
            columns = df.columns
            for column in columns:
                if how == 'all':
                    if sum(df[column].isna()) == rows:
                        goals.append(column)
                elif how == 'any':
                    if sum(df[column].isna()) != 0:
                        goals.append(column)
            return df.drop(columns = goals)

        else:
            return df
        
    except Exception as e:
        print(e)


# In[14]:



def replace(df, columns, value = 'mean'):
    try:
        for column in columns:   
            if value == 'mean':   
                goal = df[column].mean()
            elif value == 'median':
                goal = df[column].median()
            elif value == 'mode':
                goal = df[column].mode()[0]

            for i in range(len(df[column])):
                if pandas.isnull(df.loc[i, column]):
                    df.loc[i, column] = goal
        return df
    except Exception as e:
        print(e)


# In[4]:


def replaceRegression(df, X, y):
    try:
        indexes = [i for i in range(len(df)) if not pd.isnull(df.loc[i, X])]
        X_train = np.array((df.loc[indexes, X])).reshape(-1,1)
        y_train = np.array((df.loc[indexes, y])).reshape(-1,1)
        reg = LinearRegression().fit(X_train, y_train)
        for i in range(len(df)):
            if pd.isnull(df.loc[i, X]):
                df.loc[i, X] = reg.predict(df.loc[i, y])[0][0]
        return df
    except Exception as e:
        print(e)


# In[3]:


def standartization(df, columns):
    try:
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            for i in range(len(df[column])):
                df.loc[i, column] = df.loc[i, column] * mean / std
        return df
    except Exception as e:
        print(e)


# In[2]:


def scaling(df, columns):
    try:
        for column in columns:
            Max = max(df[column])
            Min = min(df[column])
            for i in range(len(df[column])):
                df.loc[i, column] = (df.loc[i, column] - Min) / (Max - Min)
            #for item id df:
            #    item = (item - Min) / (Max - Min)
        return df
    except Exception as e:
        print(e)

