import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold,\
     StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import sklearn.metrics


def internal_validation(df_features, df_metadata, clf, scaler,
                        splits_file, method, method_params, 
                        pattern_id_column, class_column, 
                        feature_columns, binary_output=False, **args):
    """Internal validation on a single dataset. Data are splitted into
    train and test sets according to the method chosen. The train and test
    splits are cached into the splits_file to ensure repeatability.
    
    NOTE: Any time you change the splitting method and/or the related
    parameters (method and method_params arguments) you need to manually
    delete the cached splits (splits_file) for the changes to take effect.
    
    Parameters
    ----------
    df_features: pd.DataFrame
        The datframe containing the features.
    df_metadata: pd.DataFrame
        The dataframe containing the metadata.
    clf: object
        The classifiesr object as for instance provided by scikit-learn.
    scaler: object or None
        A scaler object as for instance provided by scikit-learn. Pass 
        None for no scaling.
    splits_file: str
        Full (relative or absolute) path where the splits are cached (for
        repeatable results).
    method: str
        The method using for generating the train/test splits. Can be:
        - 'stratified-k-fold' -> stratified k-fold
        - 'stratified-shuffle-split' -> stratified shuffle split
        Note: stratification is based on the values of target_label. 
    method_params: dict
        Method-specific parameters.
        - n_splits: int
            The number of splits (or folds). Applies to 
            'stratified-k-fold' and 'stratified-shuffle-split'.
        - test_size: float
            Fraction of samples in the dataset to include in the test 
            split; the complement is assigned to the train split.
            Aplies to 'stratified-shuffle-split'.    
    pattern_id_column: str
        The name of the column that uniquely identifies each pattern 
        (i.e., case, instance, etc) in both the df_features and 
        df_metadata dataframes.
    class_column: str
        Name of the column that stores the class labels in the
        metadata dataframe.
    feature_columns: list of str
        Names of the columns that store the features in the
        df_features dataframe.
    binary_output: bool
        Whether to treat the classification problem as a binary one and
        return senistivity and specificity accordingly.
    binary_class_labels: tuple of str (2)
        The positive and negative class labels (in this order). Required 
        if binary_output = True.
    param_grid: dict (optional)
        Performs hyperparameter tuning via 5-fold cross validation on
        the parameter space defined by the keys and values of the 
        dictionary.
        
    Returns
    -------
    valid_res: np.array of float (n_splits, 1) or (n_splits, 3)
        The results of the validation. If binary_output = False the only
        column reports accuarcy; otherwise The first column reports
        accuracy, the second sensitivity and the third specificity.
        All the values are floats in [0,1].
    """
    
    #Initialise the return values
    if binary_output:
        valid_res = np.zeros(shape=(method_params['n_splits'], 3))
    else:
        valid_res = np.zeros(shape=(method_params['n_splits'], 1))
        
    
    #Set the primary key as index
    for df in [df_features, df_metadata]:
        df.set_index(keys=pattern_id_column, inplace=True)  
        
    #Sort by index to ensure repeatability
    df_features.sort_index(inplace=True)
    
    #Prepare the splits
    match method:
        case 'stratified-k-fold':
            splitter = StratifiedKFold(
                shuffle= True, n_splits = method_params['n_splits']
            )
        case 'stratified-shuffle-split':
            splitter = StratifiedShuffleSplit(
                n_splits = method_params['n_splits'], 
                test_size = method_params['test_size']
            )
        case _:
            raise Exception('Splitting method not supported')
            
    X=df_features.index
    y=df_metadata.loc[X][class_column]
    
    #Check if the train/test splits exists, if not generate them
    if os.path.isfile(splits_file):
        df_splits = pd.read_csv(filepath_or_buffer=splits_file)
    else:
        df_splits = pd.DataFrame(index=df_features.index)
        for i, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
            train_instances = df_features.index[train_idxs]
            test_instances = df_features.index[test_idxs]
            current_split_col_name = f'Split_{i}'
            df_splits[current_split_col_name] = 'unassigned'
            df_splits.loc[train_instances, 
                          current_split_col_name] = 'train' 
            df_splits.loc[test_instances, 
                          current_split_col_name] = 'test'
        
        #Create the destination folder if it doesn't exist and save the
        #splits
        splits_dst_folder = os.path.dirname(splits_file)
        if not os.path.isdir(splits_dst_folder):
            os.makedirs(splits_dst_folder)
            
        #Save the splits
        df_splits.to_csv(splits_file)
        
        
    #Read the splits and iterate through them
    splits = pd.read_csv(splits_file, index_col=pattern_id_column)
    for split_idx, split in enumerate(splits.columns):
        train_indices_on_df = df_features.index[splits[split] == 'train']
        test_indices_on_df = df_features.index[splits[split] == 'test']
        
        df_train = df_features.loc[train_indices_on_df]
        df_test = df_features.loc[test_indices_on_df]
        df_train_metadata = df_metadata.loc[train_indices_on_df]
        df_test_metadata = df_metadata.loc[test_indices_on_df]
        
        classification_report = cross_validation(
            df_train=df_train, df_test=df_test, 
            df_train_metadata=df_train_metadata, 
            df_test_metadata=df_test_metadata, clf=clf, scaler=scaler,
            pattern_id_column=pattern_id_column, 
            class_column=class_column, 
            feature_columns=feature_columns, **args)
                
        valid_res[split_idx, 0] = classification_report['accuracy']
        if binary_output:
            sens, spec = _get_sensitivity_specificity(
                classification_report, args['binary_class_labels']
            )
            valid_res[split_idx, 1] = sens
            valid_res[split_idx, 2] = spec
        
    #Reset the indices
    for df in [df_features, df_metadata]:
        df.reset_index(inplace=True)  
        
    return valid_res
    
            
def cross_validation(df_train, df_test, df_train_metadata, 
                     df_test_metadata, clf, scaler,
                     pattern_id_column, class_column, feature_columns,
                     complete_report=False, **args):
    """Estimates accuracy given train/test labels and features
    
    Parameters
    ----------
    df_train: pd.DataFrame
        The datframe containing the train data.
    df_test: pd.DataFrame
        The dataframe containing the test data.
    df_train_metadata: pd.DataFrame
        The dataframe containing the metadata of the train set.
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
    complete_report: bool
        If True a complete report is generated.
    param_grid: dict (optional)
        Performs hyperparameter tuning via 5-fold cross validation on
        the parameter space defined by the keys and values of the 
        dictionary.
    
    Returns
    -------
    classification_report: dict
        Summary of the precision, recall, F1 score for each class as
        returned by sklearn.metrics.classification_report().  
    df_complete_report: pd.DataFrame (optional)
        A dataframe containing the class labels and posterior 
        probabilities of each pattern in df_test. The columns are 
        organised as follows: 
         - 'Predicted labels' -> Predicted hard labels
         - All the other columns contain the posterior probabilities of
           each class, where the column name represents the class label.
    """
    #Set the primary key as index
    for df in [df_train, df_test, df_train_metadata, df_test_metadata]:
        if df.index.name != pattern_id_column:
            df.set_index(keys=pattern_id_column, inplace=True)
    
    #Get the train features and labels
    X_train = df_train[feature_columns]
    y_train = df_train_metadata.loc[df_train.index, class_column]
    
    #Get the test features
    X_test = df_test[feature_columns]
    
    #Apply feature normalisation if required
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    #Perform hyperparameter tuning if requested
    if 'param_grid' in args:
        tuned_clf = GridSearchCV(clf, args['param_grid'])
        tuned_clf.fit(X_train, y_train)
        clf = tuned_clf.best_estimator_
    
    #Train the model
    clf.fit(X=X_train, y=y_train)
    
    #Predict the labels in the test set
    y_pred = clf.predict(X=X_test)
    y_true = df_test_metadata.loc[df_test.index, class_column]
        
    #Reset the indices
    for df in [df_train, df_test, df_train_metadata, df_test_metadata]:
        df.reset_index(inplace=True)
        
    classification_report = sklearn.metrics.classification_report(
            y_true, y_pred, output_dict=True)
    
    retval = classification_report 
    
    if complete_report:

        #Compute posterior probabilities and add it to the report
        y_pred_proba = clf.predict_proba(X=X_test)
        df_complete_report = pd.DataFrame(data=y_pred_proba, 
                                          columns=clf.classes_)
        
        #Add hard predicted labels
        df_complete_report['Predicted labels'] = y_pred
           
        retval = classification_report, df_complete_report      
    
    return retval

def _get_sensitivity_specificity(classification_report, 
                                 binary_class_labels):
    """Get the sensitivity and specificity from a classification
    report for a binary classification problem
    
    Parameters
    ----------
    classification_report: dict
        A classification report as returned by cross_validation()
    binary_class_labels: tuple of str (2)
        The positive and negative class label, in this order
        
    Returns
    -------
    sens, spec: floats
        Sensitivity and specificity
    """
    sens = classification_report[binary_class_labels[0]]['recall']
    spec = classification_report[binary_class_labels[1]]['recall']
    return sens, spec
