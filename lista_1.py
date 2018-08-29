# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
 
import numpy as np


#metrics
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
##from imblearn.metrics import geometric_mean_score
#

# no dataset2, algumas amostras (4 ou 5) possuiam algumas características "?", as quais foram substituídas por 0
dataset = 'data_1'

# Generate a classification dataset
data = np.genfromtxt(dataset+".txt",delimiter=',',dtype='str')
print(data.shape)
                        #column
X = np.delete(data, data.shape[1]-1, axis=1) 
y = data[:,data.shape[1]-1]

y_=[]
_false=0
_true=0
for i in range(len(y)):
    if(y[i]=='false'):
        y_.append(0)
        _false+=1
    else:
        _true+=1
        y_.append(1)

     
y = np.array(y_)
X = X.astype(np.float)


#ploting histogram of classes
##import matplotlib.pyplot as plt
##import numpy as np
##plt.hist(y, bins=30)
##plt.ylabel('class');
##plt.show()
##
##input()
# Scale the variables to have 0 mean and unit variance
scaler = StandardScaler()

X = scaler.fit(X).transform(X)

accuracy = []
f1=[]
gmean=[]
auc=[]
fold=0 
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=30) 
for train, test in skfold.split(X,y):
    Xi_train = X[train]
    yi_train = y[train]

    Xi_test = X[test]
    yi_test = y[test]

    acc_subsample={}
    gmean_subsample={}
    f1_subsample={}
    auc_subsample={}

    print("fold:",fold)
    print(Xi_train.shape)
    print(Xi_test.shape)
    print(yi_train.shape)
    print(yi_test.shape)
    fold+=1
    for i in range(5,11):
        sub_X = Xi_train[:int(Xi_train.shape[0]*i/10)]
        sub_y = yi_train[:int(Xi_train.shape[0]*i/10)]

        base_model = Perceptron(max_iter=100)
        #Commun Bagging (samples bootstrap)
        #pool_classifiers = BaggingClassifier(base_model,bootstrap=True, n_estimators=100)
        #Random Subspace (features bootstrap)
        pool_classifiers = BaggingClassifier(base_model, n_estimators=100, bootstrap_features = True, max_features = 0.5)
        pool_classifiers.fit(sub_X, sub_y)
        
        y_pred = pool_classifiers.predict(Xi_test)
        acc_subsample[i/10]= np.mean(y_pred == yi_test)
        f1_subsample[i/10]= f1_score(yi_test, y_pred,average='weighted')
        #gmean_subsample[i/10]= geometric_mean_score(yi_test, y_pred, average='weighted')
        auc_subsample[i/10] = roc_auc_score(yi_test, y_pred,average='weighted')

        #fpr, tpr, thresholds = roc_curve(yi_test, y_pred)
        #auc_subsample[i/10]= auc(fpr, tpr)
        

        
    accuracy.append(acc_subsample)
    f1.append(f1_subsample)
#    gmean.append(gmean_subsample)
    auc.append(auc_subsample)

##np.save(dataset+"Perceptron_bagging_acc.py", accuracy)
##np.save(dataset+"Perceptron_bagging_f1.py", f1)
##np.save(dataset+"Perceptron_bagging_gmean.py", gmean)
##np.save(dataset+"Perceptron_bagging_auc.py", auc)
        


##accuracy = []
##fold=0 
##kfold = KFold(n_splits=10, shuffle=False, random_state=1) 
##for train, test in kfold.split(X):
##    Xi_train = X[train]
##    yi_train = y[train]
##
##    Xi_test = X[test]
##    yi_test = y[test]
##    score_subsample={}
##
##    print("fold:",fold)
##    fold+=1
##    for i in range(5,11):
##        sub_X = Xi_train[:int(Xi_train.shape[0]*i/10)]
##        sub_y = yi_train[:int(Xi_train.shape[0]*i/10)]
##        base_model = DecisionTreeClassifier()
##    #  base_model = Perceptron(max_iter=100)
##        #Commun Bagging (samples bootstrap)
##        pool_classifiers = BaggingClassifier(base_model, n_estimators=100)
##        #Random Subspace (features bootstrap)
##        pool_classifiers = BaggingClassifier(base_model, n_estimators=100,Bootstrap = True, bootstrap_features = True, max_features = 0.5)
##        pool_classifiers.fit(sub_X, sub_y)
##        
##        y_pred = pool_classifiers.predict(Xi_test)
##        accuracy_subsample[i/10]= 1 - np.mean(y_pred == yi_test)
##        
##    accuracy.append(accuracy_subsample)
##np.save("Perceptron_bagging_acc.py", accuracy)
##
##
##
