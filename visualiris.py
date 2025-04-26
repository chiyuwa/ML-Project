from sklearn import datasets
import pandas as pd
from logging_system import logging_common as Logger
from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

LOGNAME = "VisualIris"
log = Logger.get_logger(LOGNAME)

iris_datas = datasets.load_iris()
iris = pd.DataFrame(iris_datas.data, columns=['SpealLength', 'Spealwidth', 'PetalLength', 'PetalLength'])


"""
Basic description of the dataset
"""
iris.shape
log.info('First 5 rows of the dataset:\n{}'.format(iris.head()))
log.info('Descriptive statistics of the dataset:\n{}'.format(iris.describe()))


"""
Plot necessary graphs to visualize the dataset
"""
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
 
style_list = ['o', '^', 's']       # 设置点的不同形状，不同形状默认颜色不同，也可自定义
data = iris_datas.data
labels = iris_datas.target_names
cc = defaultdict(list)
 
for i, d in enumerate(data):
    cc[labels[int(i/50)]].append(d) 
p_list = []
c_list = []
 
for each in [0, 2]:
    plt.figure(figsize=(6, 6))
    for i, (c, ds) in enumerate(cc.items()):
        draw_data = np.array(ds)
        p = plt.plot(draw_data[:, each], draw_data[:, each+1], style_list[i])
        p_list.append(p)
        c_list.append(c)
    plt.legend([x[0] for x in p_list], c_list)
    plt.title('petail length and width') if each==0 else plt.title('speal length and width')
    plt.xlabel('length of petail(cm)') if each==0 else plt.xlabel('length of speal(cm)')
    plt.ylabel('width of petail(cm)') if each==0 else plt.ylabel('width of speal(cm)')
    if each==0:
        plt.savefig('./figs/iris_{}.png'.format('petail length and width'))
    else:
        plt.savefig('./figs/iris_{}.png'.format('speal length and width'))

