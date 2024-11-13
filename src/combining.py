import pandas as pd

def soft_late_fusion(dfs_train, df_test, df_train_metadata, 
                     df_test_metadata, clf, scaler,
                     pattern_id_column, class_column, feature_columns,
                     fusion_method='product', **args):
    """Combination of a posteriori probabilities
    
        Parameters
    ----------
    dfs_train: list of pd.DataFrame
        The datframes containing the train data.
    df_test: pd.DataFrame
        The dataframe containing the test data.
    df_train_metadata: pd.DataFrame
        The dataframe containing the metadata of the train sets.
    df_test_metadata: pd.DataFrame
        The dataframe containing the metadata of the test set.
    clf: object
        The classifiesr object as for instance provided by scikit-learn
    scaler: object or None
        A scaler object as for instance provided by scikit-learn. Pass 
        None for no scaling.
    pattern_id_column: str
        The name of the column that uniquely identifies each pattern 
        (i.e., case, instance, etc) in all the dataframes (primary key).
    class_column: str
        Name of the column that stores the class labels in the
        train and test metadata dataframes.
    feature_columns: list of str
        Names of the columns that store the features in
        df_train and df_test dataframes.
    fusion_method: str
        The method used for a posteriori fusion of class probabilities.
        Can be:
            - `product` -> product rule
            - `sum` -> sum rule
            - `max` -> max rule
    param_grid: dict (optional)
        Performs hyperparameter tuning via 5-fold cross validation on
        the parameter space defined by the keys and values of the 
        dictionary.
    """

def concatenate_features(dfs_features, feature_columns, pattern_id_column):
    """Concatenates features (early fusion)
    
    Parameters
    ----------
    dfs_features: list of pd.DataFrame (N)
        The dataframes containing the features that we want to 
        concatenate.
    feature_columns: list of list of str (N)
        For each dataframe in dfs_features the column names representing
        the features.
    pattern_id_column: str
        The name of the column that uniquely identifies each pattern 
        (i.e., case, instance, etc). Must be the same in all of the
        dataframes in dfs_features.
        
    Returns
    -------
    df_concatenated: pd.DataFrame
        A new dataframe containing the concatenated features.
    feature_columns_out: list of str
        The column names representing the features in df_concatenated 
    """
    
    def _rename_features(df_features, feature_columns, index):
        df_features_out = df_features.copy()
        
        feature_columns_out = [f'{feature_column}__{index}' for 
                               feature_column in feature_columns]
        mapper = {f_in: f_out for (f_in, f_out) in 
                  zip(feature_columns, feature_columns_out)} 
        
        df_features_out = df_features_out.rename(columns=mapper)
        return df_features_out, feature_columns_out
    
    if len(dfs_features) != len(feature_columns):
        raise Exception('There must be one feature column list for'
                        ' each dataframe')
    
    feature_columns_out = list()
    
    #Get the first dataframe
    index = 0
    df_concatenated = dfs_features[index]
    df_concatenated, new_feature_columns = _rename_features(
        df_concatenated, feature_columns[index], index)
    feature_columns_out.extend(new_feature_columns)
    
    #Join the other dataframes
    for df_features, feature_cols in zip(dfs_features[1::], 
                                         feature_columns[1::]):
        index+=1
        df_features, new_feature_columns = _rename_features(
            df_features=df_features, 
            feature_columns=feature_cols, 
            index=index)
        df_concatenated = pd.merge(df_concatenated, df_features, 
                                   on=pattern_id_column)
        feature_columns_out.extend(new_feature_columns)
    
    return df_concatenated, feature_columns_out
    
    
