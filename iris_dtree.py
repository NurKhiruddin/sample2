from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import seaborn as sns
iris = sns.load_dataset('iris')

X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)



clf.fit(Xtrain, ytrain) 
st.write((tree.plot_tree(clf.fit(Xtrain, ytrain) ))

#st.header("Mean accurary is")
#st.write(clf.score(Xtest, ytest))

