# Check the versions of libraries
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
from pandas.plotting import scatter_matrix #导入散点图矩阵包
import matplotlib.pyplot as plt
from sklearn import model_selection  #模型比较和选择包
from sklearn.metrics import classification_report  #将主要分类指标以文本输出
from sklearn.metrics import confusion_matrix #计算混淆矩阵，主要来评估分类的准确性
from sklearn.metrics import accuracy_score #计算精度得分
from sklearn.linear_model import LogisticRegression #线性模型中的逻辑回归
from sklearn.tree import DecisionTreeClassifier #树算法中的决策树分类包
from sklearn.neighbors import KNeighborsClassifier #导入最近邻算法中的KNN最近邻分类包
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #判别分析算法中的线性判别分析包
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯中的高斯朴素贝叶斯包
from sklearn.svm import SVC  #支持向量机算法中的支持向量分类包

#load dataset
url = "iris.data" #"http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names 对应数据里的花萼长、花萼宽、花瓣长、花瓣宽和Iris花类别5个字段
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#读取iris.data这个csv格式的数据
dataset = pandas.read_csv(url, names=names)

#shape - 首先是对数据的维度（样本量、变量个数等）进行了解
print(dataset.shape)

#head - 其次详细考察数据本身, 显示前20个数据
print(dataset.head(20))

#descriptions - 第三是数据所有属性的描述性统计
print(dataset.describe())

#class distribution - 第四是不同Iris花类别的数据细分
print(dataset.groupby('class').size())

#数据的可视化呈现
#box and whisker plots - 箱形图
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histogram - 直方图
dataset.hist()
plt.show()

#scatter plot matrix - 多变量图（图矩阵）
scatter_matrix(dataset)
plt.show()

#重点地方到了，机器学习开始发挥作用了。开始用算法进行评估
#这部分包括：
# 1、对数据集进行分离（分为训练集、验证集等）；
# 2、采用10倍交叉验证设置测试机制；
# 3、根据Iris鸢尾花测量维度构建5种不同模型来预测其种类；
# 4、选择最佳模型

#Split-out validation dataset - 建立验证数据集，目的是寻找我们所建立的模型中的最优者
#因此我们需要一部分与机器学习算法独立的数据集来进行评估，以此判断我们使用的评估模型的准确性
array = dataset.values #将数据库转换成数组形式
X = array[:,0:4] #取前四列，即属性数值
Y = array[:,4] #取最后一列，种类
validation_size = 0.20 #验证集规模
seed = 7
#分割数据集在；X_train and Y_train就是训练集，X_validation and Y_validation就是验证集
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#test options and evaluation metric
#测试机制：采用10倍交叉验证来评估模型的准确性。通过把我们的数据集分为10份，其中9份用于训练，1份用于验证，并重复训练分组的所有组合
seed = 7
#设置scoring变量对所构建的每个模型进行评估，其中accuracy用以评估模型的一个度量值，
# 它等于模型正确预测实际数据的数量/数据集中所有数据总数，这一比率乘以100%（比如95%的精确度）
scoring = 'accuracy'

#Spot check algorithms - 检验算法，找到最佳算法
# Spot Check Algorithms
models = [] #建立列表
#往maodels添加元组（算法名称，算法函数）
models.append(('LR', LogisticRegression())) #逻辑回归-LR
models.append(('LDA', LinearDiscriminantAnalysis())) #线性判别分析-LDA
models.append(('KNN', KNeighborsClassifier())) #K最邻近-KNN
models.append(('CART', DecisionTreeClassifier())) #分类和回归树-CART
models.append(('NB', GaussianNB())) #高斯朴素贝叶斯-NaiveBayes-NB
models.append(('SVM', SVC())) #支持向量机-SVM
# evaluate each model in turn - 按顺序依次评估上述算法
results = []
names = []
for name, model in models: #将算法名称与函数分别读取
    # 建立10倍交叉验证
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    # 每一个算法模型作为其中的参数，计算每一模型的精度得分
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#compare algorithms
# 绘制模型评估结果的图形，并比较每个模型的差异和平均精度
# 每个算法有一个精确度量的群体，因为每个算法被评估10次（10次交叉验证）
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
print("在得到KNN算法为测试中最佳模型的基础上，了解验证集上模型的准确性")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train) #knn拟合序列集
predictions = knn.predict(X_validation) #预测验证集
print(accuracy_score(Y_validation, predictions)) #验证集精度得分
print(confusion_matrix(Y_validation, predictions)) #混淆矩阵
print(classification_report(Y_validation, predictions)) #分类预测报告

#make predictions on validation dataset
print("在得到SVM算法为测试中最佳模型的基础上，了解验证集上模型的准确性")
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
