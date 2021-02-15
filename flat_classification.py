from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

################### Configuration Section #####################
path = './datasets/filtered_canada.csv'
n_folds = 5
random_state = 0
classifier = RandomForestClassifier(n_estimators=150, criterion='gini')
resampler = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42, n_jobs=4)
resampling = False
metric = 'f1-score'
###############################################################

# Loading the CSV Data
data_frame = pd.read_csv(path)
classes = data_frame['class']

# Gathering the unique classes
unique_classes = np.unique(classes)

# Configuring the Stratified K-Fold procedure
folds_result_list = []
kfold = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state = random_state)
kfold_count = 1


#Split Inputs and Outputs
input_data = data_frame.iloc[:, :-1].values
output_data = data_frame.iloc[:, -1].values

# Final results data_frame
metric_data_frame = pd.DataFrame()

frames = list()

#Run Cross-validation
for train_index, test_index in kfold.split(input_data, output_data):
    print('----------Started fold {} ----------'.format(kfold_count))
    # Slice training and testing datasets
    input_data_train, output_data_train = input_data[train_index], output_data[train_index]
    input_data_test, output_data_test = input_data[test_index], output_data[test_index]

    #Flat Resample the training fold
    if resampling is True:
        [input_data_train, output_data_train] = resampler.fit_resample(input_data_train, output_data_train)

    #Train the classifier
    classifier.fit(input_data_train, output_data_train)

    # Predict using unseen samples
    predicted = classifier.predict(input_data_test)

    #Calculating the performance of the classifier
    report = classification_report(output_data_test, predicted, target_names=unique_classes, zero_division='warn', output_dict=True)

    metric_data_frame = pd.DataFrame.from_dict(report, orient='columns')
    frames.append(pd.DataFrame(metric_data_frame))

    kfold_count += 1

    # Pipeline result being appended to the list of results for each fold
    folds_result_list.append(report)

concat_df = pd.concat(frames)
concat_df = concat_df.loc[metric, :]
print(concat_df)
print('Mean Accuracy: {}'.format(concat_df['accuracy'].mean()))
print('Mean Macro Avg F1-Score: {}'.format(concat_df['macro avg'].mean()))

