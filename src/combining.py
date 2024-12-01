import pandas as pd

def concatenate_features(dfs_features, feature_columns, 
                         pattern_id_column, **kwargs):
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
    weights: list of float (N, optional)
        The weights to be assigned to each feature set. Each feature of
        each feature set is multiplied by the corresponding weight.
        
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
    
    #Assign unit weights if not provided
    if 'weights' in kwargs:
        weights = kwargs['weights']
    else:
        weights = [1.0] * len(dfs_features)  
        
    #Multiply the features by the given weights
    for df_features, feature_columns_, weight in\
        zip(dfs_features, feature_columns, weights):
        
        for feature_column in feature_columns_:
            df_features[feature_column] =\
                df_features[feature_column] * weight 
    
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
    
    
