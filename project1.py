#!/usr/bin/env python3

import pandas as pd
import numpy as np


def read_data():
    data = pd.read_csv('DataTraining.csv', delim_whitespace = False)


    # miss = data.dropna(axis = 1)
    # miss.info()


    #################################
    #
    # full = data.dropna(subset = data.columns.values[:-3])
    # # data = data.diff()
    # # print(data.isnull())
    #
    #
    #
    # ###nan is the rows which miss some data and nan already remove the rows which miss two or two more items.
    # idx1 = data.index[:]
    # idx2 = full.index[:]
    # nan = data.loc[idx1.difference(idx2)]
    # nan = nan.dropna(thresh = 20, subset = data.columns.values[:-3])
    #
    # for column in ['schooling', 'day_of_week']:
    #     missing = missing_columns(nan, column)
    #     # print('Logistic Regression: ', sum(abs(y_test == y_predict))/y_predict.shape[0])
    #     missing_column = data_retrieve(full.drop([column, 'id', 'profit', 'responded'],axis = 1), full[column].values, missing.drop([column, 'id', 'profit', 'responded'], axis = 1))
    #     missing[column] = missing_column
    #     full = pd.concat([full, missing], axis = 0)
    #
    #
    # ######Recover custAge via linear Regression
    # from sklearn.model_selection import train_test_split
    # size = full.shape[0]
    # missing_age = missing_columns(nan, 'custAge')
    # x, _ = dummy_standardarize(pd.concat([full.drop(['custAge','id', 'profit', 'responded'], axis =1 ), missing_age.drop(['custAge','id', 'profit', 'responded'], axis =1 )]), [13,14,15,16])
    # full_x = x[:size, :]
    # x_pre = x[size:, :]
    # from sklearn.linear_model import Lasso
    #
    # regr = Lasso(alpha = 0.1)
    #
    # regr.fit(full_x, full['custAge'])
    #
    # age_pre = regr.predict(x_pre)
    #
    # missing_age['custAge'] = age_pre
    #
    # full = pd.concat([full, missing_age], axis = 0)
    #
    #
    # #####Get profit and responded
    # profit = full['profit'].values
    # responded = 1 * (full['responded'].values == 'yes')
    # #print(sum(responded))
    # full = full.drop(['id', 'profit', 'responded'], axis = 1)
    # #dummy and standardarization
    # normalized_data, scaler = dummy_standardarize(full, [0, 14, 15, 16, 17])


    # sklearn process data
    data.info()
    p = data['profit'].values
    r = 1 * (data['responded'].values == 'yes')
    #print(sum(responded))
    data = data.drop(['id', 'profit', 'responded'], axis = 1)
    values = {'custAge': 0, 'schooling': 'no', 'day_of_week': 'weekend'}
    fill = data.fillna(value = values)
    fill.info()
    from sklearn.preprocessing import LabelBinarizer, StandardScaler
    x = None
    estimator = []
    for column in fill.columns.values:
        if fill[column].dtype == object:
            le = LabelBinarizer()
            # print(fill[column])
            le.fit(fill[column].values.reshape(-1, 1))

            encode_data = le.transform(fill[column].values.reshape(-1, 1))
            if x is None:
                x = encode_data
            else:
                x = np.c_[x, encode_data]
            estimator.append(le)
        else:
            scaler = StandardScaler()
            scaler.fit(fill[column].values.reshape(-1, 1))
            encode_data = scaler.transform(fill[column].values.reshape(-1, 1))
            if x is None:
                x = encode_data
            else:
                x = np.c_[x, encode_data]
            estimator.append(scaler)

    return x, r, p, estimator, fill.columns

def missing_columns(data, columns):
    non_missing = data.dropna(subset = [columns])
    idx1 = data.index[:]
    idx2 = non_missing.index[:]
    missing = data.loc[idx1.difference(idx2)]
    return missing


def dummy_standardarize(data, index, scaler = None):
    from sklearn.preprocessing import StandardScaler

    #print(data.values.shape)
    #print(list(np.delete(data.columns,[14, 15, 16, 17] )))
    dummied_data = pd.get_dummies(data, columns=list(np.delete(data.columns,index )) )
    if scaler == None:
        scaler = StandardScaler()
    scaler.fit(dummied_data[list(dummied_data.columns.values)[:5]])
    normalized_data = np.c_[scaler.transform(dummied_data[list(dummied_data.columns.values)[:5]]), dummied_data.drop(list(dummied_data.columns.values)[:5], axis = 1).values ]
    return normalized_data, scaler


def data_retrieve(x_train, y_train, x):
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=2)
    size = x_train.shape[0]
    x, scaler = dummy_standardarize(pd.concat([x_train, x]), [0, 13, 14, 15, 16])
    x_train = x[:size,:]
    x = x[size:, :]
    neigh.fit(x_train, y_train)
    y = neigh.predict(x)

    # from sklearn.model_selection import train_test_split
    #
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)
    #
    # neigh.fit(x_train, y_train)
    #
    # y = neigh.predict(x_test)
    #
    # print(sum(abs(y_test == y))/y.shape[0])
    return y

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
    [data_x, y, r, estimator, labels] = read_data()
    data_size = data_x.shape[0]
    positive_size = int(sum(y))
    negative_size = data_size - positive_size

    #Feature selection: VarianceThreshold
    # from sklearn.feature_selection import VarianceThreshold
    # selector = VarianceThreshold(threshold=5e-4)
    # x = selector.fit_transform(data_x)
    # # print(selector.variances_)
    # # print(x.shape)

    ##########No feature selection
    x = data_x

    #Split data
    k = 500

    index_n = random.sample(range(0, negative_size), positive_size)
    index_p = random.sample(range(negative_size, data_size) , positive_size)
    index = index_n + index_p



    #split data as train and test
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


    # #Classifier: LogisticRegression Part
    # from sklearn.linear_model import LogisticRegression
    #
    # classifier = LogisticRegression(class_weight = 'balanced', penalty='l1')
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('Logistic Regression: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: SVM part
    # from sklearn.svm import SVC
    #
    # classifier = SVC(class_weight = 'balanced')
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('SVM: ',sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: LDA part
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #
    # classifier = LinearDiscriminantAnalysis(solver='lsqr')
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print(sum(y_test==1))
    # print(sum(y_test==0))
    # print('LDA: ',sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: LDA part
    # # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    # #
    # # classifier = QuadraticDiscriminantAnalysis()
    # # classifier.fit(x_train, y_train)
    # #
    # # y_predict = classifier.predict(x_test)
    # # print(sum(abs(y_test - y_predict))/len(y_predict))
    # # print(y_test)
    # # print(y_predict)
    #
    # #Classifier: Randomforest
    # from sklearn.ensemble import RandomForestClassifier
    #
    # classifier = RandomForestClassifier()
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('Random Forest: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: Adaboost
    # from sklearn.ensemble import AdaBoostClassifier
    #
    # classifier = AdaBoostClassifier(n_estimators=40)
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('Adaboost: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: Naive Bayes
    # from sklearn.naive_bayes import GaussianNB
    #
    # classifier = GaussianNB()
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('Naive Bayes: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    #
    # #Classifier: Gradient boosting
    # from sklearn.ensemble import GradientBoostingClassifier
    #
    # classifier = GradientBoostingClassifier()
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('Gradient boosting: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: GP classifier
    # from sklearn.gaussian_process import GaussianProcessClassifier
    #
    # classifier = GaussianProcessClassifier()
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('Gaussian Process: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    # #Classifier: MLP classifier
    # from sklearn.neural_network import MLPClassifier
    #
    # classifier = MLPClassifier(hidden_layer_sizes=(20,), early_stopping=True, alpha=1)
    # classifier.fit(x_train, y_train)
    #
    # y_predict = classifier.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('MLP: ', sum(abs(y_test - y_predict))/len(y_predict))
    #
    #
    # ###KNN part
    # from sklearn.neighbors import KNeighborsClassifier
    #
    # neigh = KNeighborsClassifier(n_neighbors=2)
    #
    # neigh.fit(x_train, y_train)
    #
    # y_predict = neigh.predict(x_test)
    # print(sum(y_predict - y_test == -1)/(sum(y_test ==1 )))
    # print(sum(y_predict - y_test == 1)/(sum(y_test ==0 )))
    # print('KNN: ', sum(abs(y_test - y_predict))/len(y_predict))

    print(x.shape)
    ####test data part
    data_pre = pd.read_csv('DataPredict.csv', delim_whitespace = False, header = None)
    data_pre.columns = labels
    values = {'custAge': 0, 'schooling': 'no', 'day_of_week': 'weekend'}
    data_pre = data_pre.fillna(value = values)
    data_pre.columns = range(0, labels.values.shape[0])
    x = None
    for i in data_pre.columns:
        x_predict = estimator[i].transform(data_pre[i].values.reshape(-1, 1))
        if x is None:
            x = x_predict
        else:
            x = np.c_[x, x_predict]

    print(x.shape)
