import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import scipy.stats as st

df = pd.read_csv('docs/data_arrhythmia.csv', delimiter=';', na_values='?')

missing_data = pd.DataFrame(
    {'total null': df.isnull().sum(), 'null percentage': (df.isnull().sum() / 82790) * 100})
print(missing_data)


def detect_null_columns(data):
    null_columns = []
    for column in data.columns:
        if data[column].isnull().any():
            null_columns.append(column)
    return null_columns


def detect_zero_columns(data):
    zero_columns = []
    for i, column in enumerate(data.columns):
        if (data[column] == 0).all():
            zero_columns.append((f'Position: {i + 1}', column))
    return zero_columns


columns_with_all_null = df.columns[df.isnull().all()]
df = df.drop(columns_with_all_null, axis=1)
df = df[df['height'] <= 200]  # Will drop data if the height is more than 200.
df = df[df['weight'] <= 200]  # Will drop data if the weight is more than 200.
df.drop('J', axis=1, inplace=True)
df.to_csv('docs/cleaned_file.csv', index=False)

id_data = [i for i in range(len(df))]
df['id'] = id_data

train_dataset, validation_dataset, test_dataset = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

sample_column = ['age', 'sex', 'weight', 'qrs_duration', 'qrs', 'QRST', 'heart_rate', 'diagnosis']
for label in sample_column[:-1]:
    plt.hist(df[df["sex"] == 1][label], color='blue', label='female', alpha=0.7, density=True)
    plt.hist(df[df["sex"] == 0][label], color='red', label='male', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()

data_set = ['age', 'sex', 'qrs_duration', 'qrs', 'heart_rate', 'diagnosis']
plt.figure(figsize=(18, 9))
df[data_set].boxplot()
plt.title("Numerical variables in Desired Data Set", fontsize=20)
plt.show()

q1 = df.quantile(.25)
q3 = df.quantile(.75)
iqr = q3 - q1
df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]


def standardization(all_data):
    for column in all_data.columns:
        if column != 'diagnosis':
            all_data[column] = (all_data[column] - all_data[column].mean()) / (all_data[column].std())
    return all_data


test_start_id = len(df)
all_data = standardization(df.copy())
std_train = all_data.iloc[:test_start_id]
