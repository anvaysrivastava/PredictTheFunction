# PredictTheFunction

## Intro
I am trying to play and setup tensor flow.
In this problem, given two CSVs `test.csv` and `train.csv` which contain value for some function that has not been coded in predict.py. The code is expected to estimate the value for unknown feature.

The csv are of format

| a        | b           | y  |
| ------------- |:-------------:| -----:|
| 63      | 51 | 162 |
| 76      | 94 |   98 |

## How to run
`python predict.py`

The evaluation is presently hardcoded.

## System values
python = 2.7

pip = 10.0.1

tensorflow = 1.8.0

pandas = 0.23.0

## Sample output
**python predict.py**

{'average_loss': 0.106394805, 'global_step': 10000, 'loss': 9.965647}

f(a=5,b=5)=[9.993236]

f(a=0,b=1)=[-3.003212]
