import numpy as np
import pandas as pd
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path,'train.csv')
    test_file_path = os.path.join(raw_data_path,'test.csv')
    #read data with default parameters
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path,index_col='PassengerId')
    test_df['Survived']=-888 ##create column and give default value
    
    df = pd.concat((train_df,test_df),axis=0) #merge the two datasets
    return df

def process_data(df):
    #using method chaining
    return (df
           #create title attribute
            .assign(Title = lambda x: x.Name.map(get_title))
            #working missing values
            .pipe(fill_missing_values)
            #create fare bin feature
            .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels = ['very_low','low','high','very_high']))
            #create ageState
            .assign(AgeState = lambda x: np.where(x.Age>=18, 'Adult','Child'))
            .assign(FamilySize = lambda x: x.Parch + x.SibSp +1)
            .assign(IsMother = lambda x: np.where (((x.Sex == 'female') & (x.Parch >0) &(x.Age > 18) & (x.Title != 'Miss')),1,0))
            #create Deck Feature
            .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))
            .assign(Deck = lambda x: x.Cabin.map(get_deck))
            #feature encoding
            .assign(IsMale = lambda x: np.where(x.Sex == 'male',1,0))
            .pipe(pd.get_dummies, columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])
            ##drop unnecessary columns
            .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1)
            #reorder columns
            .pipe(reorder_columns)           
           )

def get_title(name):
    title_group = {'mr': 'Mr',
                  'mrs':'Mrs',
                   'miss':'Miss',
                   'master':'Master',
                   'don':'Sir',
                   'rev':'Sir',
                   'dr':'Officer',
                   'mme':'Mrs',
                   'ms':'Mrs',
                   'major':'Officer',
                   'lady':'Lady',
                   'sir':'Sir',
                   'mlle':'Miss',
                   'col':'Officer',
                   'capt':'Officer',
                   'the countess':'Lady',
                   'jonkheer':'Sir',
                   'dona':'Lady'
                  }
    first_name_with_title = name.split(',')[1] ## get everything after comma
    title = first_name_with_title.split('.')[0]##get everything before dot
    title = title.strip().lower() ##make lower case and strip white space
    return title_group[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def fill_missing_values(df):
    #embarked
    df.Embarked.fillna('C',inplace=True)
    #fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    #age
    median_age = df.groupby('Title').Age.transform('median')
    df.Age.fillna(median_age, inplace = True)
    return df

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir,'data','processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')
    
    #train data
    df.loc[df.Survived != -888].to_csv(write_train_path)
    #test data
    columns = [column for column in df.columns if column != 'Survived']
    df.loc[df.Survived == -888, columns].to_csv(write_test_path)


if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
