import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ["a","b","y"]
def load_data(y_name='y'):
    train_path, test_path = "train1.csv", "test1.csv"

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


(train_x, train_y), (test_x, test_y) = load_data()
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

estimator = tf.estimator.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,10])

estimator.train(input_fn=lambda:train_input_fn(train_x,train_y,100), steps=10000)
metrics = estimator.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, 100))
print(metrics)
eval_x = {'a':[5,0],'b':[5,1]}
prediction = estimator.predict(input_fn=lambda:eval_input_fn(eval_x,labels=None,batch_size=100))
predictions = list(prediction)
for i in range(0,eval_x['a'].__len__()):
    print('f(a={},b={})={}'.format(eval_x['a'][i], eval_x['b'][i], predictions[i]['predictions']))
