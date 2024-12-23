import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold,\
     StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import sklearn.metrics

from ml_routines.src.combining import concatenate_features

def generate_train_test_splits(df_metadata, splits_file,
                               split_method, split_method_params, 
                               pattern_id_column, class_column):
    """Generates train and test splits for internal validation.
    
    Parameters
    ----------
    df_metadata: pd.DataFrame
        The dataframe containing the metadata.
    splits_file: str
        Full (relative or absolute) path where the splits are cached (for
        repeatable results).
    split_method: str
        The method used for generating the train/test splits. Can be:
        - 'stratified-k-fold' -> stratified k-fold
        - 'stratified-shuffle-split' -> stratified shuffle split
        Note: stratification is based on the values of class_column.
    split_method_params: dict
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
        Name of the column that identifies the class labels in the
        metadata dataframe.
        
    Returns
    -------
    df_splits: pd.DataFrame
        The dataframe cotaining the splits. The columns are organised
        as follows:
        - `{pattern_id_column}` stores the pattern ids
        - `Split_0`, `Split_1`,...,`Split_n` store the splits. Each entry
          can be `train` or `test`, respectively indicating that the
          corresponding pattern belongs to the train or test set.
    """
    
    #Set the primary key as index
    df_metadata.set_index(keys=pattern_id_column, inplace=True)  
        
    #Sort by index to ensure repeatability
    df_metadata.sort_index(inplace=True)
    
    #Prepare the splits
    match split_method:
        case 'stratified-k-fold':
            splitter = StratifiedKFold(
                shuffle= True, n_splits = split_method_params['n_splits']
            )
        case 'stratified-shuffle-split':
            splitter = StratifiedShuffleSplit(
                n_splits = split_method_params['n_splits'], 
                test_size = split_method_params['test_size']
            )
        case _:
            raise Exception('Splitting method not supported')
                
    #Check if the train/test splits exists, if not generate them
    if os.path.isfile(splits_file):
        df_splits = pd.read_csv(filepath_or_buffer=splits_file)
    else:
        df_splits = pd.DataFrame(index=df_metadata.index)
        
        #Target labels (y) and dummy features (X)
        y=df_metadata[class_column]
        X=np.zeros(shape=[len(y),1])
        
        for i, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
            train_instances = df_metadata.index[train_idxs]
            test_instances = df_metadata.index[test_idxs]
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
        
    #Reset the primary key
    df_metadata.reset_index(inplace=True)    
        
    return pd.read_csv(splits_file, index_col=pattern_id_column)

def internal_validation_combined(dfs_features, df_metadata, clf, scaler,
                                 fusion_method, splits_file, split_method,
                                 split_method_params, pattern_id_column,
                                 class_column, feature_columns_list,
                                 binary_output, **kwargs):
    """Internal validation on multiple feature sets. Data are split 
    into train and test sets according to the method chosen. The train and 
    test splits are cached into the splits_file to ensure repeatability.
    
    NOTE: Any time you change the splitting method and/or the related
    parameters (method and method_params arguments) you need to manually
    delete the cached splits (splits_file) for the changes to take effect. 
    
    Parameters
    ----------
    dfs_features: a list of pd.DataFrame (N)
        The dataframes containing the features. 
    df_metadata: pd.DataFrame
        The dataframe containing the metadata.
    clf: object
        The classifiesr object as for instance provided by scikit-learn.
    scaler: object or None
        A scaler object as for instance provided by scikit-learn. Pass 
        None for no scaling.
    fusion_method: str
        The method for fusing the feaures. See cross_validation_combined()
        for possible values.
    splits_file: str
        Full (relative or absolute) path where the splits are cached (for
        repeatable results).
    split_method: str
        The method using for generating the train/test splits. Can be:
        - 'stratified-k-fold' -> stratified k-fold
        - 'stratified-shuffle-split' -> stratified shuffle split
        Note: stratification is based on the values of class_column. 
    split_method_params: dict
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
        Name of the column that identifies the class labels in the
        metadata dataframe.
    feature_columns_list: list (N) of list of str
        The column names representing the features in each of the 
        dfs_train/dfs_test entries.
    binary_output: bool
        Whether to treat the classification problem as a binary one and
        return senistivity and specificity accordingly.
    binary_class_labels: tuple of str (2) [optional]
        The positive and negative class labels (in this order). Required 
        if binary_output = True.
    param_grid: dict (optional)
        Performs hyperparameter tuning via 5-fold cross validation on
        the parameter space defined by the keys and values of the 
        dictionary.
    weights: list of float (N, optional)
        The weights to be assigned to each feature set in the fusion 
        method chosen. See also late_fusion() and concatenate_features()
        documentation for details.
    """
    
    #Initialise the return values
    if binary_output:
        valid_res = np.zeros(shape=(split_method_params['n_splits'], 3))
    else:
        valid_res = np.zeros(shape=(split_method_params['n_splits'], 1))
                 
    #Read/generate the splits and iterate through them
    splits = generate_train_test_splits(
        df_metadata=df_metadata, 
        splits_file=splits_file, split_method=split_method, 
        split_method_params=split_method_params, 
        pattern_id_column=pattern_id_column,
        class_column=class_column)
    
   
    for split_idx, split in enumerate(splits.columns):
        
        dfs_train, dfs_test = list(), list()
        
        train_indices = splits[splits[split] == 'train'].index
        test_indices = splits[splits[split] == 'test'].index
        
        for df_features in dfs_features:
            
            #Set the primary key as index
            for df in [df_features, df_metadata]:
                df.set_index(keys=pattern_id_column, inplace=True)             
            
            df_train_metadata = df_metadata.loc[train_indices]
            df_test_metadata = df_metadata.loc[test_indices]
            
            df_train = df_features.loc[train_indices]
            df_test = df_features.loc[test_indices]
            
            #Reset the indices
            for df in [df_train, df_test, df_train_metadata,
                       df_test_metadata, df_features, df_metadata]:
                df.reset_index(inplace=True)             
        
            dfs_train.append(df_train)
            dfs_test.append(df_test) 
                      
        classification_report =\
        cross_validation_combined(
            dfs_train=dfs_train, dfs_test=dfs_test, 
            df_train_metadata=df_train_metadata, 
            df_test_metadata=df_test_metadata, clf=clf, 
            scaler=scaler, pattern_id_column=pattern_id_column, 
            class_column=class_column, 
            feature_columns_list=feature_columns_list, 
            fusion_method=fusion_method, **kwargs) 
            
        valid_res[split_idx, 0] = classification_report['accuracy']
            
        if binary_output:
            sens, spec = _get_sensitivity_specificity(
                classification_report, kwargs['binary_class_labels']
            )
            valid_res[split_idx, 1] = sens
            valid_res[split_idx, 2] = spec
    
    return valid_res


def internal_validation(df_features, df_metadata, clf, scaler,
                        splits_file, split_method, split_method_params, 
                        pattern_id_column, class_column, 
                        feature_columns, binary_output=False, **kwargs):
    """Internal validation on a single feature set. Data are splitted into
    train and test sets according to the method chosen. The train and test
    splits are cached into the splits_file to ensure repeatability.
    
    NOTE: Any time you change the splitting method and/or the related
    parameters (method and method_params arguments) you need to manually
    delete the cached splits (splits_file) for the changes to take effect.
    
    Parameters
    ----------
    df_features: pd.DataFrame
        The dataframe containing the features. 
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
    split_method: str
        The method using for generating the train/test splits. Can be:
        - 'stratified-k-fold' -> stratified k-fold
        - 'stratified-shuffle-split' -> stratified shuffle split
        Note: stratification is based on the values of class_column. 
    split_method_params: dict
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
        Name of the column that identifies the class labels in the
        metadata dataframe.
    feature_columns: list of str
        Names of the columns that store the features in the
        df_features dataframe.
    binary_output: bool
        Whether to treat the classification problem as a binary one and
        return senistivity and specificity accordingly.
    binary_class_labels: tuple of str (2) [optional]
        The positive and negative class labels (in this order). Required 
        if binary_output = True.
    param_grid: dict [optional]
        Performs hyperparameter tuning via 5-fold cross validation on
        the parameter space defined by the keys and values of the 
        dictionary.
        
    Returns
    -------
    valid_res: np.array of float (n_splits, 1) or (n_splits, 3)
        The results of the validation. If binary_output = False the only
        column reports accuracy; otherwise The first column reports
        accuracy, the second sensitivity and the third specificity.
        All the values are floats in [0,1].
    """
    
    #Initialise the return values
    if binary_output:
        valid_res = np.zeros(shape=(split_method_params['n_splits'], 3))
    else:
        valid_res = np.zeros(shape=(split_method_params['n_splits'], 1))
                 
    #Read/generate the splits and iterate through them
    splits = generate_train_test_splits(
        df_metadata=df_metadata, 
        splits_file=splits_file, split_method=split_method, 
        split_method_params=split_method_params, 
        pattern_id_column=pattern_id_column,
        class_column=class_column)    
    
   
    #Set the primary key as index
    for df in [df_features, df_metadata]:
        df.set_index(keys=pattern_id_column, inplace=True)    
    
    for split_idx, split in enumerate(splits.columns):
    
        train_indices_on_df = df_features.index[splits[split] == 'train']
        test_indices_on_df = df_features.index[splits[split] == 'test']
        
        df_train_metadata = df_metadata.loc[train_indices_on_df]
        df_test_metadata = df_metadata.loc[test_indices_on_df]
        
        df_train = df_features.loc[train_indices_on_df]
        df_test = df_features.loc[test_indices_on_df]
        
        #Reset the index
        for df in [df_train, df_test, df_train_metadata, 
                   df_test_metadata]:
            df.reset_index(inplace=True)
                        
        classification_report = cross_validation(
                df_train=df_train, df_test=df_test, 
                df_train_metadata=df_train_metadata, 
                df_test_metadata=df_test_metadata, clf=clf, scaler=scaler,
                pattern_id_column=pattern_id_column, 
                class_column=class_column, 
                feature_columns=feature_columns, **kwargs)
                
        valid_res[split_idx, 0] = classification_report['accuracy']
        if binary_output:
            sens, spec = _get_sensitivity_specificity(
                classification_report, kwargs['binary_class_labels']
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
                     complete_report=False, **kwargs):
    """Performance estimation on one given set of features
    
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
        Names of the columns that identify the features in
        df_train and df_test dataframes.
    complete_report: bool
        If True a complete report is generated.
    param_grid: dict [optional]
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
         - {pattern_id_column} -> String that uniquely identifies 
                                  each pattern to classify
         - 'Predicted_label' -> The hard labels predicted 
         - All the other columns contain the posterior probabilities of
           each class, where the column name represents the class label.
    """
    #Convert the class labels to strings
    for df in [df_train_metadata, df_test_metadata]:
        df[class_column] = df[class_column].astype(str)
    
    #Set the primary key as index
    for df in [df_train, df_test, df_train_metadata, df_test_metadata]:
        df.set_index(keys=pattern_id_column, inplace=True)
    
    #Get the train features and labels
    X_train = df_train.loc[df_train.index, feature_columns]
    y_train = df_train_metadata.loc[df_train.index, class_column]
    
    #Get the test features
    X_test = df_test.loc[df_test.index, feature_columns]
    
    #Apply feature normalisation if required
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    #Perform hyperparameter tuning if requested
    if 'param_grid' in kwargs:
        tuned_clf = GridSearchCV(clf, kwargs['param_grid'])
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
        df_complete_report['Predicted_label'] = y_pred
        
        #Add pattern ids
        df_complete_report[pattern_id_column] = df_test[pattern_id_column]
           
        retval = classification_report, df_complete_report      
    
    return retval



def cross_validation_combined(dfs_train, dfs_test, df_train_metadata, 
                              df_test_metadata, clf, scaler, 
                              pattern_id_column, class_column, 
                              feature_columns_list, 
                              fusion_method, **kwargs):
    """Performance estimation on multiple sets of features.
    
    Parameters
    ----------
    dfs_train: list of pd.DataFrame (N)
        The dataframes containing the train data.
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
        The early or late feature combination method. Can be:
        
            -------------- Feature concatenation  ----------
            - `feats-concat`
            ------------------------------------------------
            
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
    weights: list of float (N, optional)
        The weights to be assigned to each feature set in the fusion 
        method chosen. See also late_fusion() and concatenate_features()
        documentation for details.
        
    Returns
    -------
    classification_report: dict
        Summary of the precision, recall, F1 score for each class as
        returned by sklearn.metrics.classification_report().
    """
    
    match fusion_method:
        case 'feats-concat':
            
            df_train, feature_columns_train = concatenate_features(
                dfs_features=dfs_train, 
                feature_columns=feature_columns_list, 
                pattern_id_column=pattern_id_column, **kwargs)
            df_test, feature_columns_test = concatenate_features(
                dfs_features=dfs_test, 
                feature_columns=feature_columns_list, 
                pattern_id_column=pattern_id_column, **kwargs)
            feature_columns = list(set(feature_columns_train) & 
                                   set(feature_columns_test))
            
            classification_report =\
                cross_validation(
                    df_train=df_train, df_test=df_test, 
                    df_train_metadata=df_train_metadata, 
                    df_test_metadata=df_test_metadata, clf=clf, 
                    scaler=scaler, pattern_id_column=pattern_id_column, 
                    class_column=class_column, 
                    feature_columns=feature_columns, 
                    complete_report=False, **kwargs)            
            
        case 'majority-voting' | 'max' | 'sum' | 'prod':
            y_pred = late_fusion(
                dfs_train=dfs_train, dfs_test=dfs_test, 
                df_train_metadata=df_train_metadata, 
                df_test_metadata=df_test_metadata, clf=clf, scaler=scaler, 
                pattern_id_column=pattern_id_column, 
                class_column=class_column, 
                feature_columns_list=feature_columns_list, 
                fusion_method=fusion_method, **kwargs)
            df_test_metadata.set_index(keys=pattern_id_column, 
                                       inplace=True)
            y_true = df_test_metadata.loc[y_pred.index, class_column]
            df_test_metadata.reset_index(inplace=True)
            
            classification_report = sklearn.metrics.classification_report(
                            y_true, y_pred, output_dict=True)  
            
        case _:
            raise Exception(f'Fusion method *{fusion_method}* '
                            f'not supported')    
    return classification_report
    
        
def late_fusion(dfs_train, dfs_test, df_train_metadata, 
                df_test_metadata, clf, scaler, pattern_id_column, 
                class_column, feature_columns_list, 
                fusion_method='product', **kwargs):
    """Supervised classification based on hard or soft late fusion.
    
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
    weights: list of float (N, optional)
        The weights to be assigned to each feature set in the fusion
        scheme selected. If fusion_method is a combination of hard class
        labels the weights apply to the hard votes associated to each 
        feature set (i.e., weighted majority voting). Otherwise, if
        fusion_method combines posterior probabilities the weights apply
        to the posterior probabilities associated to each feature set.
        If not provided unit weights are used. Weights are internally 
        normalised to sum one.
        
    Returns
    -------
    predicted_labels: pd.Series
        A pandas Series where indices represent the pattern ids and values
        the corresponding predicted labels.
    """
    
    #Assign unit weights if not provided
    if 'weights' in kwargs:
        weights = kwargs['weights']
    else:
        weights = [1.0] * len(dfs_train)
    
    #Normalise the weigths to sum one
    weights = weights/np.linalg.norm(x=weights, ord=1)
    
    #Stack the classification results obtained with each feature set
    for idx, (df_train, df_test, feature_columns) in\
        enumerate(zip(dfs_train, dfs_test, feature_columns_list)):
        
        _, complete_report =\
        cross_validation(df_train, df_test, df_train_metadata, 
                         df_test_metadata, clf, scaler,
                         pattern_id_column, class_column, 
                         feature_columns,
                         complete_report=True, **kwargs)
        complete_report[['Instance', 'Weight']] = idx, weights[idx]
        
        if idx == 0:
            complete_reports = complete_report
        else:
            complete_reports = pd.concat((complete_reports, 
                                          complete_report))
    
    class_columns = [c for c in complete_reports.columns
                     if c not in [pattern_id_column, 'Predicted_label',
                                  'Instance', 'Weight']]
    
    match fusion_method:
        case 'majority-voting':
            
            #Convert the predicted label column to a category type so
            #that all the possible classes appear in the one-hot encoding
            #that follows
            complete_reports['Predicted_label'] = pd.Categorical(
                values=complete_reports['Predicted_label'],
                categories=class_columns
            )
            
            #For each pattern get the number of votes received by each
            #target class            
            df_votes = pd.get_dummies(
                data=complete_reports[[pattern_id_column, 
                                       'Predicted_label', 'Instance',
                                       'Weight']],
                columns=['Predicted_label'], prefix='', prefix_sep='')
            
            #Weigh the votes
            df_weighted_votes = df_votes.copy()
            for class_ in class_columns:
                df_weighted_votes[class_] = \
                    df_votes[class_] * df_votes['Weight']
                
            #Sum up the weighted votes
            df_total_votes = df_weighted_votes.groupby(
                by=pattern_id_column).agg(sum)
            
            #Get the class labels with the most weighted votes                
            predicted_labels = df_total_votes[class_columns].\
                idxmax(axis=1)
            
        case 'max' | 'sum' |  'prod':
            
            match fusion_method:
                case 'max' | 'sum' :
            
                    #Weigh the posterior probabilities
                    for class_ in class_columns:
                        complete_reports[class_] = \
                            complete_reports[class_] *\
                            complete_reports['Weight'] 
                        
                case 'prod':
                    
                    #Weigh the posterior probabilities
                    for class_ in class_columns:
                        complete_reports[class_] = \
                            complete_reports[class_] **\
                            complete_reports['Weight']                    
            
            #Combined the posterior probabilities by the given method
            combined_post_proba = \
                complete_reports[[*class_columns, pattern_id_column]].\
                groupby(by=[pattern_id_column]).agg(fusion_method) 
            
            #Get the class labels with the highest combined probability
            predicted_labels = combined_post_proba.idxmax(axis=1)

        case _:
            raise Exception('Fusion method not supported')
     
    return predicted_labels            

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
