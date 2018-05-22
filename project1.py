#!/usr/bin/env python3

import pandas as pd
import numpy as np


def read_data():
    data = pd.read_csv('DataTraining.csv', delim_whitespace = False)


    # miss = data.dropna(axis = 1)
    # miss.info()

    #reindex data TODO how to find these index consisting NAN
    custAge = data['custAge']
    schooling = data['schooling']
    day_of_week = data['day_of_week']
    full_data = data.drop(['custAge', 'schooling', 'day_of_week'], axis = 1)

    data = pd.concat([custAge, schooling, day_of_week, full_data], axis=1)

    data.info()
    print(data.isnull().any(axis = 0) )
    full = data.dropna(subset = data.columns.values[:-3])
    full.info()
    # data = data.diff()
    # print(data.isnull())



    ###nan is the rows which miss some data and nan already remove the rows which miss two or two more items.
    idx1 = data.index[:]
    idx2 = full.index[:]
    nan = data.loc[idx1.difference(idx2)]
    nan = nan.dropna(thresh = 20, subset = data.columns.values[:-3])
    nan.info()



    #recover the missing data from full to repair nan


    # ########### Output
    profit = full['profit'].values
    responded = 1 * (full['responded'].values == 'yes')
    #print(sum(responded))
    full = full.drop(['id', 'profit', 'responded'], axis = 1)

    profit_nan = nan['profit'].values
    responded_nan = 1 * (nan['responded'].values == 'yes')
    #print(sum(responded))
    nan = nan.drop(['id', 'profit', 'responded'], axis = 1)

    nan.info()

    ##retrieve data
    # y_predict, neighbor = data_retrieve(full)

    # print('Logistic Regression: ', sum(abs(y_test == y_predict))/y_predict.shape[0])


    #dummy and standardarization
    normalized_data, scaler = dummy_standardarize(full, [14, 15, 16, 17])




    #print(normalized_data.shape)
    return normalized_data, responded, profit, scaler

def missing_columns(data, columns):
    non_missing = data.dropna(subset = [columns])
    idx1 = data.index[:]
    idx2 = non_missing.index[:]
    missing = data.loc[idx1.difference(idx2)]


def dummy_standardarize(data, index):
    from sklearn.preprocessing import StandardScaler

    #print(data.values.shape)
    #print(list(np.delete(data.columns,[14, 15, 16, 17] )))
    dummied_data = pd.get_dummies(data, columns=list(np.delete(data.columns,index )) )
    scaler = StandardScaler()
    scaler.fit(dummied_data[list(dummied_data.columns.values)[:4]])
    normalized_data = np.c_[scaler.transform(dummied_data[list(dummied_data.columns.values)[:4]]), dummied_data.drop(list(dummied_data.columns.values)[:4], axis = 1).values ]

    return normalized_data, scaler


def data_retrieve(x_train, y_train, x):
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=1)

    x_train = dummy_standardarize(x_train)
    x = dummy_standardarize(x)
    neigh.fit(x_train, y_train)

    y = neigh.predict(x)

    return y, neigh

def data_viz(data):
    import matplotlib.pyplot as plt

    columns = list(data.columns)
    for tag in columns[:-2]:
        attr = getattr(data, tag)
        responded_yes = attr[data.responded == 'yes'].value_counts()
        responded_no = attr[data.responded == 'no'].value_counts()
        df=pd.DataFrame({u'Yes':responded_yes, u'No':responded_no})
        df.plot(kind='barh', stacked=True)
        plt.title(tag)
        plt.ylabel(tag)
        plt.xlabel(u"responded")
        plt.show()

if __name__ == '__main__':
    import random
    [data_x, y, r, scaler] = read_data()
    data_size = data_x.shape[0]
    positive_size = int(sum(y))
    negative_size = data_size - positive_size

    #Feature selection: VarianceThreshold
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=5e-4)
    x = selector.fit_transform(data_x)
    # print(selector.variances_)
    # print(x.shape)


    #Split data
    k = 500

    index_n = random.sample(range(0, negative_size), positive_size)
    index_p = random.sample(range(negative_size, data_size) , positive_size)
    index = index_n + index_p



    #split data as train and test
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


    #Classifier: LogisticRegression Part
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(class_weight = 'balanced', penalty='l1')
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('Logistic Regression: ', sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: SVM part
    from sklearn.svm import SVC

    classifier = SVC(class_weight = 'balanced')
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('SVM: ',sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: LDA part
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    classifier = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('LDA: ',sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: LDA part
    # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    #
    # classifier = QuadraticDiscriminantAnalysis()
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(abs(y_test - y_predict))/len(y_predict))
    # print(y_test)
    # print(y_predict)

    #Classifier: Randomforest
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('Random Forest: ', sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: Adaboost
    from sklearn.ensemble import AdaBoostClassifier

    classifier = AdaBoostClassifier(n_estimators=40)
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('Adaboost: ', sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: Naive Bayes
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('Naive Bayes: ', sum(abs(y_test - y_predict))/len(y_predict))


    #Classifier: Gradient boosting
    from sklearn.ensemble import GradientBoostingClassifier

    classifier = GradientBoostingClassifier()
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('Gradient boosting: ', sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: GP classifier
    from sklearn.gaussian_process import GaussianProcessClassifier

    classifier = GaussianProcessClassifier()
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('Gaussian Process: ', sum(abs(y_test - y_predict))/len(y_predict))

    #Classifier: MLP classifier
    from sklearn.neural_network import MLPClassifier

    classifier = MLPClassifier(hidden_layer_sizes=(20,), early_stopping=True, alpha=1)
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    print('MLP: ', sum(abs(y_test - y_predict))/len(y_predict))
