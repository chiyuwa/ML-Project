# ML-Project

## Classical Machine Learning
## Step-1

```python
pip install -r requirements.txt
```

## Step-2

visualiris.py provides visualization tools for iris dataset.

run 

```
python visualiris.py
```

figs will be saved to ./figs

## Step-3

mliris.py performed traditional machine learning methods on iris dataset, including MLP SVM and RandomForest.

run 

```
python mliris.py
```

figs will be saved to ./figs, including visualized confusion matrix AUROC AUPR for each model used.

## In Context Learning
## Step-1

```python
pip install -r requirements.txt
```

## Step-2

Put your API key for SiliconFlow into ICL_classification.py(Iris Dataset) and ICL_regression.py(California House Price dataset)

## Step-3(Optional)

If you want to change the number of In Context examples in the input prompt, you can change the number of variable K in ICL_classification.py and ICL_regression.py.

## Step-4

```
python ICL_classification.py
python ICL_regression.py
```
The predicted output and true output of each LLMs will be printed and the figure for Accuracy(classification task) and MSE(regression task) of each model will be shown on.
