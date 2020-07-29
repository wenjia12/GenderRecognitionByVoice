#使用pandas包进行处理
import pandas as pd

voice_data=pd.read_csv('voice1.csv')
x=voice_data.iloc[:,:-1]
y=voice_data.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)


#数据预处理

#使用均值对缺失值进行填补
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=0, strategy = 'mean')  
x=imp.fit_transform(x) 

#对数据集进行打乱操作并以7：3的比例划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#归一化
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x_train)
x_train = scaler1.transform(x_train)
x_test = scaler1.transform(x_test)


#建模

#Logistic分类器
from sklearn.linear_model import LogisticRegression 
logistic=LogisticRegression(max_iter=10000) 
logistic.fit(x_train,y_train)
#神经网络
from sklearn.neural_network import MLPClassifier 
nn=MLPClassifier(max_iter=100000) 
nn.fit(x_train,y_train)
#SVM支持向量机
from sklearn.svm import SVC
svc=SVC(C=1, kernel='rbf', probability=True) 
svc.fit(x_train,y_train) 
#KNN算法
from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier() 
knn.fit(x_train,y_train) 


#验证和测试
from sklearn import metrics


y_train_result=logistic.predict(x_train)
print('logistic train Accuracy Score:')
print(metrics.accuracy_score(y_train_result,y_train))
y_pred=logistic.predict(x_test)
print('logistic test Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print('\n')

y_train_result=nn.predict(x_train)
print('nn train Accuracy Score:')
print(metrics.accuracy_score(y_train_result,y_train))
y_pred=nn.predict(x_test)
print('nn test Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print('\n')

y_train_result=svc.predict(x_train)
print('svc train Accuracy Score:')
print(metrics.accuracy_score(y_train_result,y_train))
y_pred=svc.predict(x_test)
print('svm test Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print('\n')

y_train_result=knn.predict(x_train)
print('knn train Accuracy Score:')
print(metrics.accuracy_score(y_train_result,y_train))
y_pred=knn.predict(x_test)
print('knn test Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print('\n')
 
from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt

plt.figure()

###############画logistic的ROC-AUC曲线########################
prob_predict_y_validation_logistic = logistic.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_logistic = prob_predict_y_validation_logistic[:, 1]  
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, predictions_validation_logistic) 
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)  
plt.plot(fpr_logistic, tpr_logistic, 'g', label='logistic = %0.2f' % roc_auc_logistic)

###############画nn的ROC-AUC曲线########################
prob_predict_y_validation_nn = nn.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_nn = prob_predict_y_validation_nn[:, 1]  
fpr_nn, tpr_nn, _ = roc_curve(y_test, predictions_validation_nn) 
roc_auc_nn = auc(fpr_nn, tpr_nn)  
plt.plot(fpr_nn, tpr_nn, 'c', label='nn = %0.2f' % roc_auc_nn) 

###############画svm的ROC-AUC曲线########################
prob_predict_y_validation_svm = svc.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_svm = prob_predict_y_validation_svm[:, 1]  
fpr_svm, tpr_svm, _ = roc_curve(y_test, predictions_validation_svm) 
roc_auc_svm = auc(fpr_svm, tpr_svm)  
plt.plot(fpr_svm, tpr_svm, 'm', label='svm = %0.2f' % roc_auc_svm)

###############画KNN的ROC-AUC曲线########################
prob_predict_y_validation_knn = knn.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_knn = prob_predict_y_validation_knn[:, 1]  
fpr_knn, tpr_knn, _ = roc_curve(y_test, predictions_validation_knn) 
roc_auc_knn = auc(fpr_knn, tpr_knn)  
plt.plot(fpr_knn, tpr_knn, 'y', label='knn = %0.2f' % roc_auc_knn) 

###############################roc auc公共设置##################################
plt.title('ROC Validation')  
plt.legend(loc='lower right')  
plt.plot([0, 1], [0, 1], 'r--')  
plt.xlim([0, 1])  
plt.ylim([0, 1])  
plt.ylabel('True Positive Rate')  
plt.xlabel('False Positive Rate') 
