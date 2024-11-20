"""Examples of combinations of feature sets based on the penguins
dataset"""
import pandas as pd
import palmerpenguins
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_routines.src.performance_estimation import\
     internal_validation_combined

#Column that identifies the pattern ids in the feature and metadata
#dataframes. This is the primary key that links features and metadata.
pattern_id_column = 'pattern_id'

#Column that identifies the class label in the metadata dataframe
class_column = 'class_label'

#Choose a classification model. Note that soft late fusion schemes (like
#product and sum rule) require the classifier to return posterior 
#probabilities, hence probability=True.
clf = SVC(random_state=0, probability=True)

#Choose a feature scaling (normalisation) method
scaler = StandardScaler()

#Read the data and drop records containing NaN
df_penguins = palmerpenguins.load_penguins()
df_penguins.dropna(inplace=True)

#Extract the metadata
df_metadata = pd.DataFrame(
    data={pattern_id_column: df_penguins.index,
          class_column: df_penguins['species']}
)

#Define the two sets of features that will be combined
features_1 = ['bill_length_mm', 'bill_depth_mm']
features_2 = ['flipper_length_mm', 'body_mass_g']
df_features_1 = df_penguins[features_1]
df_features_2 = df_penguins[features_2]
for df in [df_features_1, df_features_2]:
    df[pattern_id_column] = df.index

#Train/test splitting settings
split_method = 'stratified-k-fold'
n_folds = 5

#Save the splits here for reproducible results (delete the file
#to generate new splits or any time n_folds changes)
splits_file_k_fold = '../src/examples/penguins/splits_file_k_fold.csv'

#Schemes for feature combination
combination_schemes = ['early-fusion', 'majority-voting', 'prod']

for combination_scheme in combination_schemes:
    accuracy = \
        internal_validation_combined(
            dfs_features=[df_features_1, df_features_2], 
            df_metadata=df_metadata, clf=clf, scaler=scaler,
            fusion_method=combination_scheme, 
            splits_file=splits_file_k_fold, split_method=split_method,
            split_method_params={'n_splits': n_folds}, 
            pattern_id_column=pattern_id_column,
            class_column=class_column, 
            feature_columns_list=[features_1, features_2],
            binary_output=False
        ) 
    
    print(f'Accuracy over the {n_folds} folds ({combination_scheme}): ')
    for accuracy_ in accuracy.flatten():
        print(f'{100*accuracy_:3.2f}%')   
    print()