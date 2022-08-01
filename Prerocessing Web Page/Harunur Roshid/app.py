from cProfile import label
from calendar import c
import csv
from pydoc import allmethods
from tabnanny import check
from tkinter import N
from unicodedata import name
#from fileinput import filename
from flask import Flask, render_template, request;
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads,ALL,DATA
from sklearn import tree
import sklearn;
from werkzeug.utils import secure_filename

import os

app = Flask(__name__)

#configuration
files= UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST']='static'
configure_uploads(app,files)

#libary
import jinja2
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.impute import SimpleImputer

from sklearn import model_selection
# regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR 
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score,classification_report

# classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preprocess")
def preprocess():
    return render_template("preprocess.html")


@app.route("/new",methods=['GET','POST'])
def new():
    if request.method== 'POST' and 'csv' in  request.files:
        file=request.files['csv']
        drop_col=request.form.get("drop")
        label1=request.form.get("label")
        label2=request.form.get("label2")
        filename=secure_filename(file.filename)
        file.save(os.path.join('static',filename))
        if filename.endswith(".csv"):
            df=pd.read_csv(os.path.join('static',filename))
            df_table=df
        else:
            df=pd.read_excel(os.path.join('static',filename))
            df_table=df

        if label2:

            label_data=df.groupby([label2]).sum()
            lenth=len(label_data)
            if lenth>2:
                check="Regression"
                print(check)
            else:
                check="Classification"
                print(check)
        else:
            check="cluster"
            print(check)

        
        first=request.form.get("first")
        last=request.form.get("last")
        select_col=request.form.get("select")
        # df_info=df.isnull.sum().sum()
        # if(df_info != 0):
        #column range select
        if not first:
            print("no select")
        else:
            df_table = df_table.iloc[:, int(first):int(last)]
        le=LabelEncoder()
        # label=request.form.get("label")
        # label1=request.form.get("label1")

        #drop selctor
        if not drop_col:
            print("empty")
        else:
            drop_col=int(drop_col)
            df_table.drop(df_table.columns[drop_col], axis=1,inplace=True)
        #auto encoding
        filcol = df_table.dtypes[df_table.dtypes == np.object]
        listcol = list(filcol.index)
        for value in listcol:
            df_table[value] = le.fit_transform(df_table[value].astype(str))

        #encoding sector

        # if label and not label1:
        #     df_table[label] = le.fit_transform(df_table[label].astype(str))
        #     print("empty")
        # elif label1:
        #     df_table[label] = le.fit_transform(df_table[label].astype(str))
        #     df_table[label1] = le.fit_transform(df_table[label1].astype(str))
        #     print("nost select")
        # else:
        #     # df_table[label] = le.fit_transform(df_table[label].astype(str))
        #     # df_table[label1] = le.fit_transform(df_table[label1].astype(str))
        #     print('nal')

        #null value handing
        df_null=df_table.isnull().sum().sum()
        if( df_null !=0):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer = imputer.fit(df_table)
            df_table = imputer.transform(df_table)
        else:
            print("no null")
        
        #normalize data
        # import scaler library and normalize x data
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0,1))
        # x = scaler.fit_transform(df_table)
        #save fil
        file_name=request.form.get("file_name")
        df_table.to_csv(file_name,index=False)
    return render_template("preprocess_com.html",df_table=df_table, df_info=df_null,check=check)
@app.route("/datauploads", methods=['GET','POST'])
def datauploads():
    if request.method== 'POST' and 'csv_data' in  request.files:
        file=request.files['csv_data']
        # check=request.form.get('method')
        train=request.form.get("input")
        label=request.form.get("label")
        checkbox=request.form.getlist('checked')
        filename=secure_filename(file.filename)
        file.save(os.path.join('static',filename))
        if filename.endswith(".csv"):
            df=pd.read_csv(os.path.join('static',filename))
            df_table=df
        else:
            df=pd.read_excel(os.path.join('static',filename))
            df_table=df
        
        label_data=df.groupby([label]).sum()
        lenth=len(label_data)
        if lenth>2:
            check="Regression"
            print(check)
        else:
            check="Classification"
            print(check)
        df_size= df.size
        df_info= df.isnull().sum().sum()
        if(df_info!=0):
            print("null")

        df_shape=df.shape
        df_head=df.head(10)
        df_column=list(df.columns)
        seed = 0

        #x and y saparate
        # x = df.iloc[:, 1:label].values
        # y = df.iloc[:, label].values           
        x = df.drop(label,axis=1)
        y = df[label]
        x_column=list(x.columns)

        #train test split
        if train:
            train= int(train)/100
            x_train,x_test,y_train,y_test= model_selection.train_test_split(x,y, test_size=train, random_state=seed)
            x1=x_train.shape
            y1=y_train.shape
        else:
            x1=x.shape
            y1=y.shape
            print("not train test select")

        #if(filename=="*.csv")

        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0,1))
        # x = scaler.fit_transform(x)
        
        models=[]
        results=[]
        names=[]
        name1=[]
        allmethods=[]
        al=allmethods.sort()
        
        scoring='accuracy'
        if check=="Regression":
            names.append(check)
            if train:
                if 'Rfr' in checkbox: 
                    rfr=RandomForestRegressor(n_estimators=100, random_state=0)
                    rfr.fit(x_train,y_train)
                    y_pred=rfr.predict(x_test)
                    rfr_score=r2_score(y_test,y_pred)
                    allmethods.append(rfr_score)
                    name1.append(rfr)
                else:
                    print("not select RandomForestRegressor")

                if "Dtr" in checkbox:
                    Dtr=DecisionTreeRegressor()
                    Dtr.fit(x_train,y_train)
                    y_pred=Dtr.predict(x_test)
                    Dtr_score=r2_score(y_test,y_pred)
                    allmethods.append(Dtr_score)
                    name1.append(Dtr)
                else:
                    print("not select DecisionTreeRegressor")

                if 'Knr' in checkbox:
                    knn=KNeighborsRegressor(n_neighbors = 10)
                    knn.fit(x_train,y_train)
                    y_pred=knn.predict(x_test)
                    knn_score=r2_score(y_test,y_pred)
                    allmethods.append(knn_score)
                    name1.append(knn)
                else:
                    print("not select KNeighborsRegress")

                if 'Lda' in checkbox:
                    lda=LinearDiscriminantAnalysis()
                    lda.fit(x_train,y_train)
                    y_pred=lda.predict(x_test)
                    lda_score=r2_score(y_test,y_pred)
                    allmethods.append(lda_score)
                    name1.append(lda)
                else:
                    print("not select LinearDiscriminantAnalysis")



                if 'Gnb' in checkbox:
                    nb= GaussianNB()
                    nb.fit(x_train,y_train)
                    y_pred=nb.predict(x_test)
                    nb_score=r2_score(y_test,y_pred)
                    allmethods.append(nb_score)
                    name1.append(nb)
                else:
                    print("not select GaussianNB")

                if 'Svr' in checkbox:
                    sv= SVR()
                    sv.fit(x_train,y_train)
                    y_pred=sv.predict(x_test)
                    sv_score=r2_score(y_test,y_pred)
                    allmethods.append(sv_score)
                    name1.append(sv)
                else:
                    print("not select SVR")





            else:
                if 'Rfr' in checkbox: 
                    rfr=RandomForestRegressor(n_estimators=100, random_state=0)
                    rfr.fit(x,y)
                    rfr_score=round(rfr.score(x,y)*100, 2)
                    allmethods.append(rfr_score)
                    name1.append(rfr)
                else:
                    print("not select RandomForestRegressor")

                if "Dtr" in checkbox:
                    Dtr=DecisionTreeRegressor()
                    Dtr.fit(x,y)
                    Dtr_score=round(Dtr.score(x,y)*100, 2)
                    allmethods.append(Dtr_score)
                    name1.append(Dtr)
                else:
                    print("not select DecisionTreeRegressor")
                
                if 'Knr' in checkbox:
                    knn=KNeighborsRegressor(n_neighbors = 10)
                    knn.fit(x,y)
                    knn_score=round(knn.score(x,y)*100, 2)
                    allmethods.append(knn_score)
                    name1.append(knn)
                else:
                    print("not select KNeighborsRegress")
                
                if 'Lda' in checkbox:
                    lda=LinearDiscriminantAnalysis()
                    lda.fit(x,y)
                    lda_score=round(lda.score(x,y)*100, 2)
                    allmethods.append(lda_score)
                    name1.append(lda)
                else:
                    print("not select LinearDiscriminantAnalysis")
                
                if 'Gnb' in checkbox:
                    nb= GaussianNB()
                    nb.fit(x,y)
                    nb_score=round(nb.score(x,y)*100, 2)
                    allmethods.append(nb_score)
                    name1.append(nb)
                else:
                    print("not select GaussianNB")

                if 'Svr' in checkbox:
                    sv= SVR()
                    sv.fit(x,y)
                    sv_score=round(sv.score(x,y)*100, 2)
                    allmethods.append(sv_score)
                    name1.append(sv)
                else:
                    print("not select SVR")
                




                

            # models.append(('RNF',RandomForestRegressor(n_estimators= 100, random_state = 0)))
            # models.append(('CART',DecisionTreeRegressor()))
            # models.append(('kNN', KNeighborsRegressor(n_neighbors = 10)))
            # models.append(('LDA',LinearDiscriminantAnalysis()))
            # models.append(('NB',GaussianNB()))
            # models.append(('SVM', SVR()))
            print("regression select")
        else:
            if not train:
                if 'Dtc' in checkbox:
                    models.append(('tree',DecisionTreeClassifier()))
                else:
                    print("not select DecisionTreeClassif")
                if 'Rfc' in checkbox:   
                    models.append(('RN', RandomForestClassifier()))
                else:
                    print("not select RandomForestClassifier")
                if 'Svc' in checkbox: 
                    models.append(('SVM', SVC()))
                else:
                    print(" not select SVC")
                if 'Knc' in checkbox:
                    models.append(('ne',KNeighborsClassifier()))
                else:
                    print("not select  KNeighborsClassifier")
                print("classification select")
                for name, model in models:
                    kfold=model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
                    cv_results= np.mean(model_selection.cross_val_score(model,x,y,cv=kfold,scoring=scoring))

                    results.append(cv_results)
                    names.append(name)
                    msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    allmethods.append(msg)
                    model_results= results
                    model_names= names
                    print(msg)

            else:
                if 'Dtc' in checkbox:
                    tree=DecisionTreeClassifier()
                    tree.fit(x_train,y_train)
                    y_pred=tree.predict(x_test)
                    tree_score=accuracy_score(y_test,y_pred)
                    allmethods.append(tree_score)
                    name1.append(tree)
                    print(classification_report(y_test, y_pred))
                else:
                    print("not select DecisionTreeClassif")
                if 'Rfc' in checkbox:    
                    rfc=RandomForestClassifier()
                    rfc.fit(x_train,y_train)
                    y_pred=rfc.predict(x_test)
                    rfc_score=accuracy_score(y_test,y_pred)
                    allmethods.append(rfc_score)
                    name1.append(rfc)
                    print(classification_report(y_test, y_pred))
                else:
                    print("not select RandomForestClassifier")
                if 'Svc' in checkbox: 
                    sv=SVC()
                    sv.fit(x_train,y_train)
                    y_pred=sv.predict(x_test)
                    sv_score=accuracy_score(y_test,y_pred)
                    allmethods.append(sv_score)
                    name1.append(sv)
                    print(classification_report(y_test, y_pred))
                else:
                    print(" not select SVC")
                if 'Knc' in checkbox:
                    knn=KNeighborsClassifier()
                    knn.fit(x_train,y_train)
                    y_pred=knn.predict(x_test)
                    knn_score=accuracy_score(y_test,y_pred)
                    allmethods.append(knn_score)
                    name1.append(knn)
                    print(classification_report(y_test, y_pred))
                else:
                    print("not select  KNeighborsClassifier")





                


            

      
    return render_template("details.html", filename=filename,
                                      df_table=df,input=input,label=label,
                                      df_size=df_size,df_info=df_info,
                                      df_shape=df_shape,df_column=df_column,
                                        df_head=df_head, x_column=x_column,y_column=y,
                                        x_train=x1,y_train=y1,check=check,
                                        model_names= names, model_results=allmethods,name=name1)
                                        




if __name__=="__main__":
    app.run(debug=True)

