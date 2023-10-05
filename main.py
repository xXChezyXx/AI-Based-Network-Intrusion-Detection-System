import math
import os
import re
import sys
import gzip
from enum import Enum
from io import StringIO
from time import time
import numpy
import pandas
import tensorflow
import requests
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

PRNG_SEED = 42
DATASET_COLUMNS_FILE = os.path.join("dataset", "kddcup1999_columns.txt")
DATASET_URL = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
DATASET_FILE = os.path.join("dataset", "kddcup1999.csv")

if not os.path.exists(DATASET_COLUMNS_FILE):
    with requests.get("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names") as request:
        with open(DATASET_COLUMNS_FILE, 'wb') as file:
            file.write(request.content)

if not os.path.exists(DATASET_FILE):
    with requests.get(DATASET_URL) as response:
        if response.status_code != 200:
            raise RuntimeError(f"failed to download dataset: {DATASET_URL}")
        with open(DATASET_FILE, 'wb') as file:
            file.write(gzip.decompress(response.content))

numpy.random.seed(PRNG_SEED)
tensorflow.random.set_seed(PRNG_SEED)

ColumnType = Enum('ColumnType', 'SYMBOLIC CONTINUOUS')
column_types = {}

with open(DATASET_COLUMNS_FILE, 'r') as file:
    column_labels: str = file.read()

column_regex: re.Pattern = re.compile(r"^(?P<column_name>\w+): (?P<data_type>\w+)\.$")
for column_type in column_labels.splitlines()[1:]:
    match = column_regex.match(column_type)
    column_types[match.group("column_name")] = ColumnType[match.group("data_type").upper()]

dataframe = pandas.read_csv(
    DATASET_FILE,
    header=None,
)

dataframe.columns = [*column_types.keys(), "outcome"]

unique_dataframe = dataframe.drop_duplicates()

encoded_dataframe = pandas.get_dummies(
    unique_dataframe,
    columns=[column_name for column_name, column_type in column_types.items() if column_type == ColumnType.SYMBOLIC],
    drop_first=True,
)

TRAINING_PROPORTION = .25
TESTING_PROPORTION = .70
VALIDATION_PROPORTION = .05

assert TRAINING_PROPORTION + TESTING_PROPORTION + VALIDATION_PROPORTION <= 1.0

partitions = {'train': pandas.DataFrame()}

remaining_dataframe = pandas.DataFrame()
grouped_outcomes = encoded_dataframe.groupby('outcome')
maximum_per_class = math.ceil((TRAINING_PROPORTION * len(grouped_outcomes.groups['normal.'])) / (len(grouped_outcomes.groups.keys()) - 1))
for key in grouped_outcomes.groups.keys():
    group = grouped_outcomes.get_group(key)

    training, remaining = train_test_split(
        group,
        shuffle=True,
        train_size=min(math.ceil(TRAINING_PROPORTION * len(group.index)), maximum_per_class) if key != "normal." else TRAINING_PROPORTION,
    )
    if key != "normal.":
        training = training.sample(n=maximum_per_class, replace=True, random_state=PRNG_SEED)
    partitions['train'] = partitions['train']._append(training)
    remaining_dataframe = remaining_dataframe._append(remaining)

partitions['test'], partitions['validate'] = train_test_split(
    remaining_dataframe,
    shuffle=True,
    train_size=TESTING_PROPORTION / (1 - TRAINING_PROPORTION),
    test_size=VALIDATION_PROPORTION / (1 - TRAINING_PROPORTION),
)

pandas.set_option("mode.chained_assignment", None)

for column_name, column_type in column_types.items():
    if column_type == ColumnType.CONTINUOUS:
        mean = partitions['train'][column_name].mean()
        std = partitions['train'][column_name].std()
        if std == 0:
            std = 1

        for partition_name, dataframe in partitions.items():
            dataframe.loc[:,column_name] = (dataframe[column_name] - mean) / std

tf_input = {
    partition_name: dataframe.drop("outcome", axis="columns").reset_index(drop=True)
    for partition_name, dataframe in partitions.items()
}

tf_output = {
    partition_name: pandas.DataFrame(
        [
            [1, 0] if outcome == 'normal.' else [0, 1]
            for outcome in dataframe[['outcome']].to_numpy()
        ],
        columns=['normal', 'intrusion'],
    )
    for partition_name, dataframe in partitions.items()
}

model = Sequential(
    [
        Dense(128, input_dim=tf_input['train'].shape[1]),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(tf_output['train'].shape[1], activation='softmax')
    ],
    name="kdd_cup_1999_classification"
)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

start_time = time()
original_stdout = sys.stdout
sys.stdout = captured_stdout = StringIO()

tf_input['train'] = numpy.asarray(tf_input['train']).astype('float32')
tf_input['validate'] = numpy.asarray(tf_input['validate']).astype('float32')
tf_output['train'] = numpy.asarray(tf_output['train']).astype('int32')
tf_output['validate'] = numpy.asarray(tf_output['validate']).astype('int32')

try:
    if not os.path.exists("temp/model_best_weights.hdf5"):
        model.fit(
            tf_input['train'], tf_output['train'],
            validation_data=(tf_input['validate'], tf_output['validate']),
            callbacks=[
                EarlyStopping(monitor='val_loss', min_delta=.001, patience=100, mode='auto', verbose=0),
                ModelCheckpoint(filepath="temp/model_best_weights.hdf5", save_best_only=True, verbose=0)
            ],
            epochs=1000,
            verbose=2,
        )
finally:
    sys.stdout = original_stdout
    
print(f"Model trained in {time() - start_time:.6f} seconds")

start_time = time()
model.load_weights("temp/model_best_weights.hdf5")
predicted = numpy.argmax(model.predict(tf_input['test'].astype('float32')), axis=1)
true_output = numpy.argmax(tf_output['test'].to_numpy().astype('float32'), axis=1)

print(f"Predicted {len(predicted):,} classifications in {time() - start_time:.6f} seconds")
print(classification_report(true_output, predicted, digits=4, target_names=tf_output['test'].columns))