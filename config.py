
# if true, only a very small dataset will be used.
# Good for running pipelines fast while developing/debugging
flag_use_reduced_dataset = False
reduced_dataset_size = 0.05

#several constants
y_column = 'went_on_backorder'
proj_random_state = 22

# sources
csv_file_train = 'input/Training_Dataset_v2.csv'
csv_file_test = 'input/Test_Dataset_v2.csv'

# folders cache
cache_dir_complete_dataset = 'cache/'
cache_dir_reduced_dataset = 'cache_reduced/'

# folders output

output_dir_complete_dataset = 'output/'
output_dir_reduced_dataset = 'output_reduced/'

cache_dir = cache_dir_reduced_dataset if flag_use_reduced_dataset else cache_dir_complete_dataset
output_dir = output_dir_reduced_dataset if flag_use_reduced_dataset else output_dir_complete_dataset

# 01_data_understanding
df_initial = cache_dir + 'df_initial.h5'
df_test_initial = cache_dir + 'df_test_initial.h5'

# 02_data_cleaning
df_cleaned = cache_dir + 'df_cleaned.h5'
df_test_cleaned = cache_dir + 'df_test_cleaned.h5'

# 03_new_features
df_new_features = cache_dir + 'df_new_features.h5'
df_test_new_features = cache_dir + 'df_test_new_features.h5'

# 04_features_selections
df_features_selections = cache_dir + 'df_features_selection.h5'
df_test_features_selections = cache_dir + 'df_test_features_selection.h5'


