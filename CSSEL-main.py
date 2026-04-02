from collections import Counter
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from smote_variant import KNNOR_SMOTE
from smote_variant import SMOTE_IPF
from smote_variant import ROSE
from smote_variant import A_SUWO
from smote_variant import SOMO
#from imbens.sampler import ADASYN, BorderlineSMOTE
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from RKNN import RKNN
from resampled import resampled
from imblearn.over_sampling import SMOTE, SMOTEN
from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
class CSSEL:
    def fit(self, X_train, y_train,X_test,n_nbor,modelk,modelna):
        self._X_train = X_train
        self._y_train = y_train
        X_train, y_train = resampled(X_train, y_train)
        X_train = pd.DataFrame(X_train)
        X = np.array(X_train)
        y_train = pd.DataFrame(y_train)
        majority_class_count = y_train[y_train.iloc[:, 0] == 0].shape[0]
        minority_class_count = y_train[y_train.iloc[:, 0] == 1].shape[0]
        X_test = pd.DataFrame(X_test)
        correlation_matrix = np.corrcoef(X, rowvar=False)
        binary_correlation_matrix = np.where(correlation_matrix >= 0, 1, 0)
        uniques = np.unique(binary_correlation_matrix, axis=0)
        uniques = uniques[[not np.all(uniques[i] == 0) for i in range(uniques.shape[0])], :]
        print(uniques)
        Result= pd.DataFrame([])
        j=0
        for y in uniques:
            l = []
            for i, x in enumerate(y):
                if x == 1:
                    l.append(i)
            print("特征子集")
            print(l)
            X_test1=X_test.iloc[:, l]
            X_train1 = X_train.iloc[:, l]
            y_trainx = pd.DataFrame(y_train)
            l_rknn = list((RKNN(X_train1, y_trainx, n_nbors=n_nbor)))
            result = pd.value_counts(l_rknn)
            l_r=len(l_rknn)
            ss1 = result.tolist()
            ss2 = result.index.tolist()
            lk_s = []
            for i in range(len(ss1)):
                lk_s.append(ss2[i])
            if (majority_class_count-l_r) >= minority_class_count:
                for i in lk_s:
                    X_train1.drop(index=i, inplace=True)
                    y_trainx.drop(index=i, inplace=True)
            else:
                lk_x=lk_s[:(majority_class_count-minority_class_count)]
                for i in lk_x:
                    X_train1.drop(index=i, inplace=True)
                    y_trainx.drop(index=i, inplace=True)
            print("Counter of y_resampled")
            y_c = np.array(y_trainx)
            print(Counter(y_c.flatten()))
            if (modelk == "DecisionTree"):
                clft = DecisionTreeClassifier()
            elif (modelk == "RandomForest"):
                clft = RandomForestClassifier()
            elif (modelk == "GradientBoosting"):
                clft = GradientBoostingClassifier()
            elif (modelk == "SVC"):
                clft = SVC()
            elif (modelk == "XGB"):
                clft = XGBClassifier()
            elif (modelk == "LogR"):
                clft = LogisticRegression()
            elif (modelk == "MLP"):
                clft = MLPClassifier()
            elif (modelk == "ExtraTreeClassifier"):
                clft = ExtraTreeClassifier()
            elif (modelk == "KNeighborsClassifier"):
                clft = KNeighborsClassifier()
            elif (modelk == "GaussianNB"):
                clft = GaussianNB()
            elif (modelk == "AdaBoostClassifier"):
                clft = AdaBoostClassifier()
            if (modelna == "SMOTE"):
                lclf = SMOTE()
            #elif (modelna == "ADASYN"):
                #lclf = ADASYN()
            elif (modelna == "SMOTEN"):
                lclf = SMOTEN()
            #elif (modelna == "BorderlineSMOTE"):
                #lclf = BorderlineSMOTE()
            elif (modelna == "KNNOR_SMOTE"):
                lclf = KNNOR_SMOTE()
            elif (modelna == "A_SUWO"):
                lclf = A_SUWO()
            elif (modelna == "ROSE"):
                lclf =ROSE()
            elif (modelna == "SMOTE_IPF"):
                lclf =SMOTE_IPF()
            elif (modelna == "SOMO"):
                lclf = SOMO()
            X_train1=np.array(X_train1)
            y_trainx=np.array(y_trainx).flatten()
            X_resampled, y_resampled = lclf.fit_resample(X_train1, y_trainx)
            clft.fit(X_resampled,y_resampled)
            y_pred1=clft.predict(X_test1)
            y_p=y_pred1.tolist()
            Result.loc[:,j]=y_p
            j=j+1
        modes = [Result.iloc[i].mode()[0] for i in range(len(Result))]
        return (modes)
dataset = pd.read_csv('data/ecoli1.csv', encoding='utf-8', delimiter=",")
dataset = np.array(dataset)
X = dataset[:, 1:-1]
y = dataset[:, -1]
kf = StratifiedKFold(n_splits=2, shuffle=True)
for train_index, test_index in kf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    lclf=CSSEL()
    y_pred=lclf.fit(X_train,y_train,X_test,modelk="SVC",modelna="SOMO",n_nbor=5)
    print(y_pred)