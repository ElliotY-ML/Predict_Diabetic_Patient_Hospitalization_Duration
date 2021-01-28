import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    generic_match = []
    
    for d in df['ndc_code']:
        if pd.isna(d):
            generic_match.append(np.nan)
        else:
            generic_match.append(ndc_df['Non-proprietary Name'][ndc_df['NDC_Code']==d].values[0])
    
    df['generic_drug_name'] = generic_match
    
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    sort_df = df.sort_values(['patient_nbr', 'encounter_id'])
    first_encounter_list = sort_df['encounter_id'][~sort_df.duplicated(subset=['patient_nbr']).values].values
    
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_list)]
     
    return first_encounter_df
    
#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    
    permutate_df = df.iloc[np.random.permutation(len(df))]
    unique_values = permutate_df[patient_key].unique()
    total_values = len(unique_values)
    train_size = round(0.6 * total_values)
    val_size = round(0.8 * total_values)
    
    train = permutate_df[permutate_df[patient_key].isin(unique_values[:train_size])].reset_index(drop = True)
    validation = permutate_df[permutate_df[patient_key].isin(unique_values[train_size:val_size])].reset_index(drop = True)
    test = permutate_df[permutate_df[patient_key].isin(unique_values[val_size:])].reset_index(drop = True)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        pre_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key = c, vocabulary_file = vocab_file_path, num_oov_buckets = 1)
        
        tf_categorical_feature_column = tf.feature_column.embedding_column(pre_categorical_feature_column, dimension = 1000)
        
        output_tf_list.append(tf_categorical_feature_column)
        
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return ((col - mean)/std)



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std = STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key = col, normalizer_fn = normalizer, default_value=default_value, dtype = tf.float64)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = (df[col] >=5).replace({True:1, False:0})
    return student_binary_prediction
