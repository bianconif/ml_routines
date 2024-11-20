"""Example of internal validation based on the iris dataset"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_routines.src.performance_estimation import internal_validation

#Column that identifies the pattern ids in the feature and metadata
#dataframes. This is the primary key that links features and metadata.
pattern_id_column = 'pattern_id'

#Column that identifies the class label in the metadata dataframe
class_column = 'class_label'

#Choose a classification model
clf = SVC(random_state=0)

#Choose a feature scaling (normalisation) method
scaler = StandardScaler()

#Load the iris dataset
data = load_iris(return_X_y=False, as_frame=True)

#Get the feature values and names 
df_features = data.data
df_features[pattern_id_column] = df_features.index
feature_names = data.feature_names

#Create the metadata dataframe
df_metadata = pd.DataFrame({class_column: data.target,
                            pattern_id_column: data.target.index})

#=======================================================================
#=== Accuracy estimation based on stratified 5-fold cross-validation ===
#=======================================================================

#Train/test splitting settings
split_method = 'stratified-k-fold'
n_folds = 5

#Save the splits here for reproducible results (delete the file
#to generate new splits or any time n_folds changes)
splits_file_k_fold = '../src/examples/iris/splits_file_k_fold.csv'

accuracy = internal_validation(
    df_features=df_features, df_metadata=df_metadata, 
    clf=clf, scaler=scaler, splits_file=splits_file_k_fold, 
    split_method=split_method, 
    split_method_params={'n_splits': n_folds}, 
    pattern_id_column=pattern_id_column, class_column=class_column, 
    feature_columns=feature_names)

print(f'Accuracy over the {n_folds} folds ({split_method}): ')
for accuracy_ in accuracy.flatten():
    print(f'{100*accuracy_:3.2f}%')
#=======================================================================
#=======================================================================
#=======================================================================

#=======================================================================
#=== Accuracy estimation based on stratified stratified shuffle ========
#=== split validation ==================================================
#=======================================================================

#Train/test splitting settings
split_method = 'stratified-shuffle-split'
n_splits = 10

#Fraction of samples to be assigned to the test set in each split
test_size = 0.3

#Save the splits here for reproducible results (delete the file
#to generate new splits or any time either of n_splits or test_size 
#changes)
splits_file_ss = '../src/examples/iris/splits_file_ss.csv'

accuracy = internal_validation(
    df_features=df_features, df_metadata=df_metadata, 
    clf=clf, scaler=scaler, splits_file=splits_file_ss, 
    split_method=split_method, 
    split_method_params={'n_splits': n_splits, 'test_size': test_size}, 
    pattern_id_column=pattern_id_column, class_column=class_column, 
    feature_columns=feature_names)

print(f'Accuracy over the {n_splits} splits ({split_method}): ')
for accuracy_ in accuracy.flatten():
    print(f'{100*accuracy_:3.2f}%')
#=======================================================================
#=======================================================================
#=======================================================================

