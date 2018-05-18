#!/usr/bin/env python3

import pandas as pd
import numpy as np


def read_data():
    data = pd.read_csv('DataTraining.csv', delim_whitespace = False)
    # data.info()
    # null_columns=data.columns[data.isnull().any()]
    # data = data[data.notnull()]
    # nan = data.dropna()
    # nan.info()
    # # data = data.diff()
    # # print(data.isnull())
    #
    # data.info()
    # print(null_columns)
    # data_viz(data)
    profit = data['profit'].values
    responded = 1 * (data['responded'].values == 'yes')
    #print(sum(responded))
    data = data.drop(['id', 'profit', 'responded'], axis = 1)

    data.info()
    # nan = data.dropna(thresh = 18)
    # nan.info()
    # nan = data.dropna(thresh = 19)
    # nan.info()
    # nan = data.dropna(thresh = 20)
    # nan.info()
    # nan = data.dropna(thresh = 21)
    # nan.info()
    nan = data.dropna()
    nan.info()
    # data = data.diff()
    # print(data.isnull())

    data.info()

    left = data - nan

    from sklearn.preprocessing import StandardScaler

    #print(data.values.shape)
    #print(list(np.delete(data.columns,[14, 15, 16, 17] )))
    dummied_data = pd.get_dummies(data, columns=list(np.delete(data.columns,[14, 15, 16, 17] )) )
    scaler = StandardScaler()
    scaler.fit(dummied_data[list(dummied_data.columns.values)[:4]])
    normalized_data = np.c_[scaler.transform(dummied_data[list(dummied_data.columns.values)[:4]]), dummied_data.drop(list(dummied_data.columns.values)[:4], axis = 1).values ]
    #print(normalized_data.shape)
    return normalized_data, responded, profit


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
    [data_x, y, r] = read_data()
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
