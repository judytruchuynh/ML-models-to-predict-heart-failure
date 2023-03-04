#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



from scipy.stats import skew, kurtosis

def descriptive_stats(data):
    mean = data.mean()
    median = data.median()
    std = data.std()
    var = data.var()
    min_value = data.min()
    percentile25 = data.quantile(.25)
    percentile50 = data.quantile(.50)
    percentile75 = data.quantile(.75)
    max_value = data.max()
    skew_value = skew(data)
    kurtosis_value = kurtosis(data)
    describe_data={"mean":mean,"median":median,"std":std,"var":var,"min":min_value,"25%":percentile25,"50%":percentile50,"75%":percentile75,"max":max_value,"skew":skew_value,"kurtosis":kurtosis_value}
    df_describe = pd.DataFrame(describe_data)
    return df_describe 

#****************************************************************************************


def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
# CRUCIAL IMPORTANT


def Visualize_confusion_matrix(y_test,y_pred_test):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm=confusion_matrix(y_test,y_pred_test)
    cm_df=pd.DataFrame(cm)
    plt.figure(figsize=(2.5,2))
    sns.heatmap(cm_df, annot=True, fmt="g", cmap="viridis")
    plt.title("Accuracy:{0:.3f}".format(accuracy_score(y_test,y_pred_test)))
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.show()
    return


from sklearn.metrics import classification_report





# Fit a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
def Naive_Bayers_Classifier(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train) 
    y_pred_test = model.predict(X_test) 
    Visualize_confusion_matrix(y_test, y_pred_test)   
    reportNB=classification_report(y_test,y_pred_test)
    print(reportNB)
    return


# # Fit a Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
def Logistic_Regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train) 
    y_pred_test = model.predict(X_test) 
    Visualize_confusion_matrix(y_test, y_pred_test)   
    reportLR=classification_report(y_test,y_pred_test)
    print(reportLR)
    return


# # Fit a support vector machine (SVM) model



from sklearn.svm import SVC
def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train, y_train) 
    y_pred_test = model.predict(X_test) 
    Visualize_confusion_matrix(y_test, y_pred_test)   
    reportSVM=classification_report(y_test,y_pred_test)
    print(reportSVM)
    return


# # Fit a random forest classifier
from sklearn.ensemble import RandomForestClassifier
def Random_Forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)     
    Visualize_confusion_matrix(y_test, y_pred_test)   
    reportRF=classification_report(y_test,y_pred_test)
    print(reportRF)
    return
###### THE RESULT CHANGES IN DIFFERENT RUNNING TIMES #############


# # Fit a k-nearest neighbors (KNN) model
from sklearn.neighbors import KNeighborsClassifier
def K_nearest_neighbors(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)   
    Visualize_confusion_matrix(y_test, y_pred_test)   
    reportKNN=classification_report(y_test,y_pred_test)
    print(reportKNN)
    return



# Fit a multi-layer perceptron (MLP) neural network
from sklearn.neural_network import MLPClassifier
def MLP_neural_network(X_train, X_test, y_train, y_test):
    model = MLPClassifier()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test) 
    Visualize_confusion_matrix(y_test, y_pred_test)   
    reportMLP=classification_report(y_test,y_pred_test)
    print(reportMLP)
    return
###### THE RESULT CHANGES IN DIFFERENT RUNNING TIMES #############




def df_compare_models(X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    technique=input("Enter technique:")
    # Initialize the models
    models = {'Naive Bayes': GaussianNB(),
              'Logistic Regression': LogisticRegression(),
              'Support Vector Machine': SVC(),
              'Random Forest': RandomForestClassifier(),
              'K-Nearest Neighbors': KNeighborsClassifier(),
              'Multi-Layer Perceptron': MLPClassifier()}
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({'Model': name, 'Technique': technique, 'Accuracy': round((accuracy),2), 'Precision': round((precision),2), 'Recall': round((recall),2), 'F1-Score': round((f1),2)})
    df_compare = pd.DataFrame(results)
    return df_compare




def plot_model_performance(dataframe, metrics):
    fig, ax = plt.subplots()
    width = 0.2
    x_ = np.arange(len(dataframe["Model"]))

    for i, metric in enumerate(metrics):
        ax.bar(x_ + i*width - (len(metrics)-1)*width/2, dataframe[metric].values, width, label=metric)

    ax.set_title("Comparison among Models' Performance")
    ax.set_xlabel("Models")
    ax.set_ylabel("Performance Metrics")
    ax.set_xticks(x_)
    ax.set_xticklabels(dataframe["Model"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.autofmt_xdate(rotation=30)
    plt.show()




def significant_features_man_chi(df, target_col):
    Mann_Whitney_significant_features = []
    Chi_Square_significant_features = []
    from scipy.stats import mannwhitneyu, chi2_contingency

    for col in df.columns:
        if col == target_col:
            continue
       
        group1 = df[df[target_col] == 0][col]
        group2 = df[df[target_col] == 1][col]
        stat, p1 = mannwhitneyu(group1, group2)  
        if p1 < 0.05:
            Mann_Whitney_significant_features.append(col)
        #a non-parametric test - determine if two independent samples were drawn from the same distribution
        
        
        contingency_table = pd.crosstab(df[target_col], df[col])
        # create a contingency table
        stat, p2, dof, expected = chi2_contingency(contingency_table)
        if p2 < 0.05:        
            Chi_Square_significant_features.append(col)
        #a statistical test - to determine relationship between target and feature variables.

    print("Mann Whitney significant features: ", Mann_Whitney_significant_features)
    print("Chi Square significant features: ", Chi_Square_significant_features)
    return




def feature_importance_grap_RandomForest(X, y):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_names = X.columns
    sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(6,4))
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importances")
    plt.title("Important features of the Random Forest Model")
    plt.show()
    return





