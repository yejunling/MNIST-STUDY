from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import time

mnist = input_data.read_data_sets('mnist-data',one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

#load data
train_X = mnist.train.images                #训练集样本
validation_X = mnist.validation.images      #验证集样本
test_X = mnist.test.images                  #测试集样本
#labels
train_Y = mnist.train.labels                #训练集标签
validation_Y = mnist.validation.labels      #验证集标签
test_Y = mnist.test.labels                  #测试集标签

# print(train_X.shape,train_Y.shape)          #输出训练集样本和标签的大小

#查看数据，例如训练集中第一个样本的内容和标签
# print(train_X[0])       #是一个包含784个元素且值在[0,1]之间的向量
# print(train_Y[0])

#可视化样本，下面是输出了训练集中前20个样本
# fig, ax = plt.subplots(nrows=4,ncols=5,sharex='all',sharey='all')
# ax = ax.flatten()
# for i in range(20):
#   img = train_X[i].reshape(28, 28)
#   ax[i].imshow(img,cmap='Greys')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

# 回归
# def fn_mean_squared(regr,x,y,option):
#   regr.fit(x, y)
#   pred = regr.predict(x)
#   # print('Coef',regr.coef_)
#   from sklearn.metrics import mean_squared_error
#   print(option, mean_squared_error(y, pred))

# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# line = LinearRegression()
# ridge = Ridge(alpha=5.01)
# lasso = Lasso(alpha=0.00001)

# fn_mean_squared(line,train_X,train_Y,'train-LinearRegression-MSE:')
# fn_mean_squared(ridge,train_X,train_Y,'train-Ridge-MSE:')
# fn_mean_squared(lasso,train_X,train_Y,'train-Lasso-MSE:')

# fn_mean_squared(line,validation_X,validation_Y,'validation-LinearRegression-MSE:')
# fn_mean_squared(ridge,validation_X,validation_Y,'validation-Ridge-MSE:')
# fn_mean_squared(lasso,validation_X,validation_Y,'validation-Lasso-MSE:')

# fn_mean_squared(line,test_X,test_Y,'test-LinearRegression-MSE:')
# fn_mean_squared(ridge,test_X,test_Y,'test-Ridge-MSE:')
# fn_mean_squared(lasso,test_X,test_Y,'test-Lasso-MSE:')


def changeData(d):
  list=[0]*len(d)
  for i in range(len(d)):
    for j in range(len(d[i])):
      if d[i][j]==1:
        list[i]=j
        break
  return list

from sklearn.metrics import accuracy_score, recall_score, f1_score
# SVM
# from sklearn.svm import SVC,LinearSVC
# models=[]
# models.append(('svm', SVC(decision_function_shape='ovo')))
# for name,clf in models:
#   print("开始时间 :", time.asctime( time.localtime(time.time()) ))
#   clf.fit(train_X,changeData(train_Y))
#   xy_lst=[(train_X,changeData(train_Y)), (validation_X,changeData(validation_Y)), (test_X,changeData(test_Y))]
#   for i in range(len(xy_lst)):
#     x_part=xy_lst[i][0]
#     y_part=xy_lst[i][1]
#     y_pred=clf.predict(x_part)
#     print(i)
#     print(name,'-ACC:',accuracy_score(y_part, y_pred))
#     print(name,'-REC:',recall_score(y_part, y_pred,average='micro'))
#     print(name,'-F1:',f1_score(y_part, y_pred,average='micro'))
#     print("结束时间 :", time.asctime( time.localtime(time.time()) ))

# NN
startTime=time.asctime( time.localtime(time.time()) )
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
mdl = Sequential()
mdl.add(Dense(50,input_dim=len(train_X[0])))
mdl.add(Activation('sigmoid'))
mdl.add(Dense(10))
mdl.add(Activation('softmax'))
sgd=SGD(lr=0.01)
# mdl.compile(loss='mean_squared_error',optimizer=sgd)
mdl.compile(loss='mean_squared_error',optimizer='adam')
mdl.fit(train_X,train_Y,nb_epoch=100,batch_size=2000)
xy_lst=[(train_X,changeData(train_Y)), (validation_X,changeData(validation_Y)), (test_X,changeData(test_Y))]
print("开始时间 :", startTime)
for i in range(len(xy_lst)):
  x_part=xy_lst[i][0]
  y_part=xy_lst[i][1]
  y_pred=mdl.predict_classes(x_part)
  print(i)
  print('NN','-ACC:',accuracy_score(y_part, y_pred))
  print('NN','-REC:',recall_score(y_part, y_pred,average='micro'))
  print('NN','-F1:',f1_score(y_part, y_pred,average='micro'))
  print("结束时间 :", time.asctime( time.localtime(time.time()) ))