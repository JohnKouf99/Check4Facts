import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")





def NaiveBayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train,y_train)
    y_pred = nb.predict(np.array(X_test))

    # #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(nb, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()



    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of Naive Bayes')
    plt.show()


def LogisticReg(X_train, X_test, y_train, y_test):

    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(np.array(X_test))

    #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(lr, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred,)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()

    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of Logistic Regression')
    plt.show()

def KNeigh(X_train, X_test, y_train, y_test):

    kn = KNeighborsClassifier()
    kn.fit(X_train,y_train)
    y_pred = kn.predict(np.array(X_test))


    #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(kn, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred,)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()

    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of K-Neighbours')
    plt.show()


def SVM(X_train, X_test, y_train, y_test):

    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)


    #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(svm_classifier, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred,)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()

    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of Support Vector Machine')
    plt.show()




def DecisionTree(X_train, X_test, y_train, y_test):

    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    y_pred = dt.predict(np.array(X_test))


    #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(dt, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred,)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()
    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of decision tree')
    plt.show()

def RandomForest(X_train, X_test, y_train, y_test):
    rf  = RandomForestClassifier()
    rf.fit(X_train,y_train)
    y_pred = rf.predict(np.array(X_test))


    #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(rf, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred,)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()

    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of Random Forest')
    plt.show()

def MLP(X_train, X_test, y_train, y_test):

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp_classifier.fit(X_train, y_train)
    y_pred = mlp_classifier.predict(X_test)

    #model evaluation
    confusion_matrix = metrics.confusion_matrix(np.array(y_test), y_pred )
    classification = metrics.classification_report(np.array(y_test), y_pred )
    scores = cross_val_score(mlp_classifier, X_train,np.array(y_train), cv=10)

    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    #print results and plot confusion matrix
    print ("Model Accuracy:", np.round(accuracy,3))
    print()
    print ("Model Recall:", np.round(recall,3))
    print()
    print ("Model Precision:", np.round(precision,3))
    print()
    print ("Model F1-Score:", np.round(f1,3))
    print()

    print ("Cross validation score:", np.round(scores,3))
    print()
    print("Classification report:" "\n", classification) 
    print()

    plt.figure(figsize=(14, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of MLP')
    plt.show()




def cross_val_Naive_Bayes(X_train, y_train, X_val, y_val, df):
    model = GaussianNB()
    sub_accs = []
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)

    y_pred = pd.Series(y_pred, index = y_val.index)

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for Naive Bayes: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs


def cross_val_LR(X_train, y_train, X_val, y_val,df):
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)

    y_pred = pd.Series(y_pred, index = y_val.index)
    sub_accs = []

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        # print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for LR: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs
        

def cross_val_Kneigh(X_train, y_train, X_val, y_val,df):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
        
    y_pred = pd.Series(y_pred, index = y_val.index)
    sub_accs = []

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        # print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for KNeigh: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs


def cross_val_RF(X_train, y_train, X_val, y_val,df):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
        
    y_pred = pd.Series(y_pred, index = y_val.index)
    sub_accs = []

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        # print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for Random Forest: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs


def cross_val_DT(X_train, y_train, X_val, y_val, df):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
        
    y_pred = pd.Series(y_pred, index = y_val.index)
    sub_accs = []

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        # print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for Decision Tree: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs


def cross_val_SVC(X_train, y_train, X_val, y_val, df):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
        
    y_pred = pd.Series(y_pred, index = y_val.index)
    sub_accs = []

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        # print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for SVM: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs


def cross_val_MLP(X_train, y_train, X_val, y_val, df):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
        
    y_pred = pd.Series(y_pred, index = y_val.index)
    sub_accs = []

    for source in df.unique():
        
        subset_idx = list(df.index[df==source])

        # print(subset_idx)
        print()
      
        sub_y_pred = y_pred.loc[subset_idx]
        sub_y_val = y_val.loc[subset_idx]

        sub_acc = metrics.accuracy_score(sub_y_val,sub_y_pred)
        print(f"Accuracy for data source {source}: {sub_acc}")
        sub_accs.append((source, sub_acc))


    # Display fold accuracy
    print(f"Accuracy for MLP: {accuracy}\n")
    print(sub_accs)
    return accuracy, sub_accs

