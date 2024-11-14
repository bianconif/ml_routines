import pandas as pd
import sklearn.metrics

from ml_routines.src.performance_estimation import cross_validation

def late_fusion(dfs_train, dfs_test, df_train_metadata, 
                df_test_metadata, clf, scaler,
                pattern_id_column, class_column, 
                feature_columns_list, fusion_method='product', 
                **args):
    """Combination of a posteriori probabilities or hard labels
    
    Parameters
    ----------
    dfs_train: list of pd.DataFrame (N)
        The datframes containing the train data.
    dfs_test: list of pd.DataFrame (N)
        The dataframes containing the test data.
    df_train_metadata: pd.DataFrame
        The dataframe containing the metadata of the train sets.
    df_test_metadata: pd.DataFrame
        The dataframe containing the metadata of the test sets.
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
    feature_columns_list: list (N) of list of str
        The column names representing the features in each of the 
        dfs_train/dfs_test entries.
    fusion_method: str
        The method used for a posteriori fusion of class probabilities.
        Can be:
            
            -------- Combination of hard class labels ------
            - `majority-voting`
            ------------------------------------------------
            
            ---- Combination of posterior probabilities ----
            - `prod` -> product rule
            - `sum` -> sum rule
            - `max` -> max rule
            ------------------------------------------------
            
    param_grid: dict (optional)
        Performs hyperparameter tuning via 5-fold cross validation on
        the parameter space defined by the keys and values of the 
        dictionary.
        
    Returns
    -------
    classification_report: dict
        Summary of the precision, recall, F1 score for each class as
        returned by sklearn.metrics.classification_report().
    """
    
    #Stack the classification results obtained with each feature set
    for idx, (df_train, df_test, feature_columns) in\
        enumerate(zip(dfs_train, dfs_test, feature_columns_list)):
        
        _, complete_report =\
        cross_validation(df_train, df_test, df_train_metadata, 
                         df_test_metadata, clf, scaler,
                         pattern_id_column, class_column, 
                         feature_columns,
                         complete_report=True, **args)
        complete_report['Instance'] = idx
        
        if idx == 0:
            complete_reports = complete_report
        else:
            complete_reports = pd.concat((complete_reports, 
                                          complete_report))
    
    class_columns = [c for c in complete_reports.columns
                     if c not in [pattern_id_column, 'Predicted_label',
                                  'Instance']]
    
    match fusion_method:
        case 'majority-voting':

            #Remember the method value_counts() returns the frequency of 
            #values in a column in decrescent order
            combined_hard_labels = \
                complete_reports[[pattern_id_column, 'Predicted_label']].\
                groupby(by=[pattern_id_column]).\
                agg(lambda x:x.value_counts().index[0]).reset_index()
            
            y_pred = combined_hard_labels['Predicted_label']
            
        case 'max' | 'sum' | 'prod':
            #Combined the posterior probabilities by the given method
            combined_post_proba = \
                complete_reports[[*class_columns, pattern_id_column]].\
                groupby(by=[pattern_id_column]).agg(fusion_method) 
            
            #Get the class labels with the highest combined probability
            y_pred = combined_post_proba.idxmax(axis=1)
        case _:
            raise Exception('Fusion method not supported')
    
    #Prepare the classification report
    df_test_metadata.set_index(keys=pattern_id_column, inplace=True)
    y_true = df_test_metadata.loc[combined_post_proba.index, 
                                  class_column]
    df_test_metadata.reset_index(inplace=True)
    classification_report = sklearn.metrics.classification_report(
                y_true, y_pred, output_dict=True)            
    
    return classification_report

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
    
    
