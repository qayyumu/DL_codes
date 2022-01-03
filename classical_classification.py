#! \usr\bin\env python
### LinearClassifier Example with perceptron training etc
###
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from mlxtend.plotting import plot_decision_regions 
from sklearn.model_selection import cross_val_score

x,Y = make_classification(n_features=2,n_classes=2,n_samples=100,n_redundant=0,n_clusters_per_class=1)
# x,Y = make_circles(n_samples=100,noise=0.03,factor=0.7)


clf_RF = RandomForestClassifier()
clf_LR = LogisticRegression()
clf_SVC = SVC() 

clf_RF.fit(x,Y)
clf_RF_acc = clf_RF.score(x,Y) 

clf_LR.fit(x,Y)
clf_LR_acc = clf_LR.score(x,Y) #max(cross_val_score(clf_LR,x,Y,cv=5,scoring='accuracy'))

clf_SVC.fit(x,Y)
clf_SVC_acc = clf_SVC.score(x,Y) #max(cross_val_score(clf_SVC,x,Y,cv=5,scoring='accuracy'))

plt.subplot(1,3,1)
plot_decision_regions(x,Y,clf_RF,legend=2)
plt.title("Random Forest: Acc= "+str(clf_RF_acc*100.) )
plt.subplot(1,3,2)
plot_decision_regions(x,Y,clf_LR,legend=2)
plt.title("Logistic Regression: Acc= " + str(clf_LR_acc*100.))
plt.subplot(1,3,3)
plot_decision_regions(x,Y,clf_SVC,legend=2)
plt.title("Support Vector Machine: Acc= "+str(clf_SVC_acc*100.))

plt.show()

