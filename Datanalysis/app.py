from cProfile import label
from calendar import c
import csv
from fileinput import filename
from pydoc import allmethods
from tabnanny import check
from tkinter import N
from turtle import st
from unicodedata import name
#from fileinput import filename
from flask import Flask, Response, redirect, render_template, request, send_file, session, url_for;
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads,ALL,DATA
from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib.figure import Figure
import pandas as pd
from sklearn import model_selection, tree
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score;
from werkzeug.utils import secure_filename
#from flask.ext.session import Session
from flask_session import Session
sess = Session()


import os
from flask import Flask
from io import BytesIO 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
from sklearn.preprocessing import MinMaxScaler

 


app = Flask(__name__)
#configuration
SESSION_TYPE = 'redis'
files= UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST']='static'
configure_uploads(app,files)
#app.config.from_object(__name__)
Session(app)

app.secret_key = os.urandom(30)
scaler=MinMaxScaler(feature_range=(0,1))
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/preprocess")
def preprocess():
    return render_template("preprocess.html")

@app.route("/datauploads", methods=['GET','POST'])
def datauploads():
    if request.method== 'POST' and 'csv_data' in  request.files:
        file=request.files['csv_data']
        # check=request.form.get('method')
        train=request.form.get("input")
        label=request.form.get("label")
        checkbox=request.form.getlist('checked')
        global filename
        filename=secure_filename(file.filename)
        file.save(os.path.join('static',filename))
        if filename.endswith(".csv"):
            global df
            df=pd.read_csv(os.path.join('static',filename))

            df_table=df
        else:
            df=pd.read_excel(os.path.join('static',filename))
            df_table=df
            
        print(label)
        df_table[['Day','Month',"Year"]]=df_table.TransDate.str.split('/',expand=True)
        df_table['Total_Day']=pd.to_numeric(df_table['Year'])*365+pd.to_numeric(df_table['Month'])*30+pd.to_numeric(df_table['Day'])
        session["unknown"] = df_table.to_json()
        df1=df.groupby("Total_Day").sum()
        df1=df1.reset_index()[['Total_Day','Amount']]

        # df1=df1[["Total_Day","Amount"]]
        x1 = df1.drop(['Amount'],axis='columns')
        y1 = df1.Amount

        #create model

        df2=df1.reset_index()['Amount']
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))
        df3=scaler.fit_transform(np.array(df2).reshape(-1,1))
        ##splitting dataset into train and test split
        training_size=int(len(df3)*0.75)
        test_size=len(df3)-training_size
        train_data,test_data=df3[0:training_size,:],df3[training_size:len(df3),:1]

        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]  
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
        
        time_step = 5
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        ### Create the Stacked LSTM model
        import tensorflow as tf
        tf.config.experimental.list_physical_devices('GPU')
        
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        #tf.config.experimental.list_physical_devices('GPU')
        # new_model=tf.keras.models.Sequential()
        # new_model.add(tf.keras.layers.LSTM(50,return_sequences=True,input_shape=(5,1)))
        # #new_model.add(tf.keras.layers.LSTM(50,return_serquences=True))
        # new_model.add(tf.keras.layers.LSTM(50))
        # new_model.add(tf.keras.layers.Dense(1))
        # new_model.compile(loss='mean_squared_error',optimizer='adam')
        # new_model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=32)
        # new_model.save('my_model.h5')
        #SSfrom tensorflow.keras import models
        new_model = tf.keras.models.load_model('my_model.h5')
        new_model.summary()
        train_predict=new_model.predict(X_train)
        test_predict=new_model.predict(X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        x_input=test_data[-5:].reshape(1,-1)
        x_input.shape
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        from numpy import array

        lst_output=[]
        n_steps=5
        i=0
        while(i<30):
    
            if(len(temp_input)>5):
        #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
                yhat =new_model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
        #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat =new_model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
    

        print(lst_output)
        day_new=np.arange(1,101)
        day_pred=np.arange(101,131)

        plt.plot(day_new,scaler.inverse_transform(df3[-100:]))
        plt.plot(day_pred,scaler.inverse_transform(lst_output))

        df4=df3.tolist()
        df4.extend(lst_output)
        print(plt.plot(df4[-50:]))
        df4=scaler.inverse_transform(df4).tolist()
        print(plt.plot(df4))
        index=df1.index[-1]
        index=index+29
        prediction=df4[index]
        print(prediction)
        


       

        # label_data=df.groupby([label]).sum()
        # lenth=len(label_data)
        # print(lenth)
        # if lenth>2:
        #     check="Regression"
        #     print(check)
        # else:
        #     check="Classification"
        #     print(check)
        global df_size, df_info,df_shape,df_column
        df_size= df.size
        df_info= df.isnull().sum().sum()
        if(df_info!=0):
            print("null")

        df_shape=df.shape
        df_head=df.head(10)
        df_column=list(df.columns)
        amount=df["Qty"]
        lnprice=np.log(amount)
        plt.plot(lnprice)





        ############################################# yearly prediction ##########################################

        df2=df.groupby("Year").sum()
        df3=df2.reset_index()[['Year','Amount']]
        x3 = df3.drop(['Amount'],axis='columns')
        y3 = df3.Amount
        df3=df3.reset_index()['Amount']
        print(df2)
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))
        df3=scaler.fit_transform(np.array(df3).reshape(-1,1))
        training_size=int(len(df3)*0.75)
        print(training_size)
        test_size=len(df3)-training_size
        train_data,test_data=df3[0:training_size,:],df3[training_size:len(df3),:1]


        # time_step = 1
        # X_train, y_train = create_dataset(train_data, time_step)
        # X_test, ytest = create_dataset(test_data, time_step)
        # print(X_train.shape)
        # print(X_test.shape)
        # X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        # X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        new_model1 = tf.keras.models.load_model('Year_model.h5')
        new_model1.summary()
        # train_predict=new_model.predict(X_train)
        # test_predict=new_model.predict(X_test)
        # train_predict=scaler.inverse_transform(train_predict)
        # test_predict=scaler.inverse_transform(test_predict)
        x_input=test_data[-1:].reshape(1,-1)
        x_input.shape
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        from numpy import array

        lst_output=[]
        n_steps=1
        i=0
        while(i<30):
    
            if(len(temp_input)>1):
        #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
                yhat = new_model1.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
        #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = new_model1.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
    

        print(lst_output)

        df5=df3.tolist()
        df5.extend(lst_output)
        value=scaler.inverse_transform(df5).tolist()
        print(value[5])









        #--------------------------------------------------------------------------------------------------------#

        
        
        

    
        
        
   
        # seed = 0

        # #x and y saparate
        # # x = df.iloc[:, 1:label].values
        # # y = df.iloc[:, label].values           
        # x = df.drop(label,axis=1)
        # y = df[label]
        # x_column=list(x.columns)

        # #train test split
        # if train:
        #     train= int(train)/100
        #     x_train,x_test,y_train,y_test= model_selection.train_test_split(x,y, test_size=train, random_state=seed)
        #     x1=x_train.shape
        #     y1=y_train.shape
        # else:
        #     x1=x.shape
        #     y1=y.shape
        #     print("not train test select")

        # #if(filename=="*.csv")

        # # from sklearn.preprocessing import MinMaxScaler
        # # scaler = MinMaxScaler(feature_range=(0,1))
        # # x = scaler.fit_transform(x)
        
        # models=[]
        # results=[]
        # names=[]
        # name1=[]
        # allmethods=[]
        # al=allmethods.sort()
        
        # scoring='accuracy'
        # if check=="Regression":
        #     names.append(check)
        #     if train:
        #         if 'Rfr' in checkbox: 
        #             rfr=RandomForestRegressor(n_estimators=100, random_state=0)
        #             rfr.fit(x_train,y_train)
        #             y_pred=rfr.predict(x_test)
        #             rfr_score=r2_score(y_test,y_pred)
        #             allmethods.append(rfr_score)
        #             name1.append(rfr)
        #         else:
        #             print("not select RandomForestRegressor")

        #         if "Dtr" in checkbox:
        #             Dtr=tree.DecisionTreeRegressor()
        #             Dtr.fit(x_train,y_train)
        #             y_pred=Dtr.predict(x_test)
        #             Dtr_score=r2_score(y_test,y_pred)
        #             allmethods.append(Dtr_score)
        #             name1.append(Dtr)
        #         else:
        #             print("not select DecisionTreeRegressor")

        #         if 'Knr' in checkbox:
        #             knn=KNeighborsRegressor(n_neighbors = 10)
        #             knn.fit(x_train,y_train)
        #             y_pred=knn.predict(x_test)
        #             knn_score=r2_score(y_test,y_pred)
        #             allmethods.append(knn_score)
        #             name1.append(knn)
        #         else:
        #             print("not select KNeighborsRegress")

        #         if 'Lda' in checkbox:
        #             lda=LinearDiscriminantAnalysis()
        #             lda.fit(x_train,y_train)
        #             y_pred=lda.predict(x_test)
        #             lda_score=r2_score(y_test,y_pred)
        #             allmethods.append(lda_score)
        #             name1.append(lda)
        #         else:
        #             print("not select LinearDiscriminantAnalysis")



        #         if 'Gnb' in checkbox:
        #             nb= GaussianNB()
        #             nb.fit(x_train,y_train)
        #             y_pred=nb.predict(x_test)
        #             nb_score=r2_score(y_test,y_pred)
        #             allmethods.append(nb_score)
        #             name1.append(nb)
        #         else:
        #             print("not select GaussianNB")

        #         if 'Svr' in checkbox:
        #             sv= SVR()
        #             sv.fit(x_train,y_train)
        #             y_pred=sv.predict(x_test)
        #             sv_score=r2_score(y_test,y_pred)
        #             allmethods.append(sv_score)
        #             name1.append(sv)
        #         else:
        #             print("not select SVR")





        #     else:
        #         if 'Rfr' in checkbox: 
        #             rfr=RandomForestRegressor(n_estimators=100, random_state=0)
        #             rfr.fit(x,y)
        #             rfr_score=round(rfr.score(x,y)*100, 2)
        #             allmethods.append(rfr_score)
        #             name1.append(rfr)
        #         else:
        #             print("not select RandomForestRegressor")

        #         if "Dtr" in checkbox:
        #             Dtr=DecisionTreeRegressor()
        #             Dtr.fit(x,y)
        #             Dtr_score=round(Dtr.score(x,y)*100, 2)
        #             allmethods.append(Dtr_score)
        #             name1.append(Dtr)
        #         else:
        #             print("not select DecisionTreeRegressor")
                
        #         if 'Knr' in checkbox:
        #             knn=KNeighborsRegressor(n_neighbors = 10)
        #             knn.fit(x,y)
        #             knn_score=round(knn.score(x,y)*100, 2)
        #             allmethods.append(knn_score)
        #             name1.append(knn)
        #         else:
        #             print("not select KNeighborsRegress")
                
        #         if 'Lda' in checkbox:
        #             lda=LinearDiscriminantAnalysis()
        #             lda.fit(x,y)
        #             lda_score=round(lda.score(x,y)*100, 2)
        #             allmethods.append(lda_score)
        #             name1.append(lda)
        #         else:
        #             print("not select LinearDiscriminantAnalysis")
                
        #         if 'Gnb' in checkbox:
        #             nb= GaussianNB()
        #             nb.fit(x,y)
        #             nb_score=round(nb.score(x,y)*100, 2)
        #             allmethods.append(nb_score)
        #             name1.append(nb)
        #         else:
        #             print("not select GaussianNB")

        #         if 'Svr' in checkbox:
        #             sv= SVR()
        #             sv.fit(x,y)
        #             sv_score=round(sv.score(x,y)*100, 2)
        #             allmethods.append(sv_score)
        #             name1.append(sv)
        #         else:
        #             print("not select SVR")
                




                

        #     # models.append(('RNF',RandomForestRegressor(n_estimators= 100, random_state = 0)))
        #     # models.append(('CART',DecisionTreeRegressor()))
        #     # models.append(('kNN', KNeighborsRegressor(n_neighbors = 10)))
        #     # models.append(('LDA',LinearDiscriminantAnalysis()))
        #     # models.append(('NB',GaussianNB()))
        #     # models.append(('SVM', SVR()))
        #     print("regression select")
        # else:
        #     if not train:
        #         if 'Dtc' in checkbox:
        #             models.append(('tree',DecisionTreeClassifier()))
        #         else:
        #             print("not select DecisionTreeClassif")
        #         if 'Rfc' in checkbox:   
        #             models.append(('RN', RandomForestClassifier()))
        #         else:
        #             print("not select RandomForestClassifier")
        #         if 'Svc' in checkbox: 
        #             models.append(('SVM', SVC()))
        #         else:
        #             print(" not select SVC")
        #         if 'Knc' in checkbox:
        #             models.append(('ne',KNeighborsClassifier()))
        #         else:
        #             print("not select  KNeighborsClassifier")
        #         print("classification select")
        #         for name, model in models:
        #             kfold=model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
        #             cv_results= np.mean(model_selection.cross_val_score(model,x,y,cv=kfold,scoring=scoring))

        #             results.append(cv_results)
        #             names.append(name)
        #             msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #             allmethods.append(msg)
        #             model_results= results
        #             model_names= names
        #             print(msg)

        #     else:
        #         if 'Dtc' in checkbox:
        #             tree=DecisionTreeClassifier()
        #             tree.fit(x_train,y_train)
        #             y_pred=tree.predict(x_test)
        #             tree_score=accuracy_score(y_test,y_pred)
        #             allmethods.append(tree_score)
        #             name1.append(tree)
        #             print(classification_report(y_test, y_pred))
        #         else:
        #             print("not select DecisionTreeClassif")
        #         if 'Rfc' in checkbox:    
        #             rfc=RandomForestClassifier()
        #             rfc.fit(x_train,y_train)
        #             y_pred=rfc.predict(x_test)
        #             rfc_score=accuracy_score(y_test,y_pred)
        #             allmethods.append(rfc_score)
        #             name1.append(rfc)
        #             print(classification_report(y_test, y_pred))
        #         else:
        #             print("not select RandomForestClassifier")
        #         if 'Svc' in checkbox: 
        #             sv=SVC()
        #             sv.fit(x_train,y_train)
        #             y_pred=sv.predict(x_test)
        #             sv_score=accuracy_score(y_test,y_pred)
        #             allmethods.append(sv_score)
        #             name1.append(sv)
        #             print(classification_report(y_test, y_pred))
        #         else:
        #             print(" not select SVC")
        #         if 'Knc' in checkbox:
        #             knn=KNeighborsClassifier()
        #             knn.fit(x_train,y_train)
        #             y_pred=knn.predict(x_test)
        #             knn_score=accuracy_score(y_test,y_pred)
        #             allmethods.append(knn_score)
        #             name1.append(knn)
        #             print(classification_report(y_test, y_pred))
        #         else:
        #             print("not select  KNeighborsClassifier")

        global user_list
        user_list=df['Brand'].unique().tolist()
    return render_template("details.html", filename=filename,
                                      df_table=df.head(),input=input,user_list=user_list,
                                      df_size=df_size,df_info=df_info,
                                      df_shape=df_shape,df_column=df_column,prediction=prediction, df7=value[5])
 
import seaborn as sns
fig,ax=plt.subplots(figsize=(6,6))
ax=sns.set_style(style="darkgrid")                                   #     df_head=df_head, x_column=x_column,y_column=y,
@app.route("/visualize")
def visualize():
    dat = session.get('unknown')
    df = pd.read_json(dat,orient='records')
    x=df['Rate']
    amon=df['Qty']
    sns.lineplot(x,amon)
    # inrate=np.log(amon)
    # a=plt.plot(inrate)
    img=BytesIO()
    canvas=FigureCanvas(fig)
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')

fig,ax=plt.subplots(figsize=(6,6))
ax=sns.set_style(style="darkgrid")    
@app.route("/visualize1")
def visualize1():
    dat = session.get('unknown')
    df = pd.read_json(dat,orient='records')
    x=df['Rate']
    amon=df['Amount']
    sns.lineplot(x,amon)
    # inrate=np.log(amon)
    # a=plt.plot(inrate)
    img=BytesIO()
    canvas=FigureCanvas(fig)
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')

fig,ax=plt.subplots(figsize=(6,6))
ax=sns.set_style(style="darkgrid")    
@app.route("/visualize3")
def visualize3():
    dat = session.get('known')
    df = pd.read_json(dat,orient='records')
    x=df['Rate']
    amon=df['Amount']
    sns.lineplot(x,amon)
    # inrate=np.log(amon)
    # a=plt.plot(inrate)
    img=BytesIO()
    canvas=FigureCanvas(fig)
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')

fig,ax=plt.subplots(figsize=(6,6))
ax=sns.set_style(style="darkgrid")    
@app.route("/visualize4")
def visualize4():
    dat = session.get('known')
    df = pd.read_json(dat,orient='records')
    x=df['Rate']
    #amon=df['Qty']
    sns.lineplot(x)
    # inrate=np.log(amon)
    # a=plt.plot(inrate)
    img=BytesIO()
    canvas=FigureCanvas(fig)
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')

fig,ax=plt.subplots(figsize=(6,6))
#ax=sns.set_style(style="darkgrid")    
@app.route("/visualize5")
def visualize5():
    fig = create_figure()
    output = BytesIO()
    #..getvalue()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    dat=session.get('nown')
    df=pd.read_json(dat, orient='records')
    print(df)
    axis.plot(df)
    return fig
    # dat=session.get('nown')
    # df=pd.read_json(dat, orient='records')
    # plt.plot(df)
    # # x=df['Rate']
    # # amon=df['Qty']
    # # plt.plot(x,amon)
    # img=BytesIO()
    # canvas=FigureCanvas(fig)
    # fig.savefig(img)
    # img.seek(0)
    # return send_file(img, mimetype='img/png')
                                     #     x_train=x1,y_train=y1,check=check,
                                    #     model_names= names, model_results=allmethods,name=name1)
@app.route("/login", methods=["POST","GET"])                                      
def login():
    dat = session.get('unknown')
    df = pd.read_json(dat,orient='records')
    print(df)
    if request.method=="POST":
        user=request.form["Brand"]
        if user!='Overall':
            global df_size, df_info,df_shape,df_column
            df = df[df['Brand'] == user]
            session["known"] = df.to_json()

            df1=df.groupby("Total_Day").sum()
            df1=df1.reset_index()[['Total_Day','Amount']]

        # df1=df1[["Total_Day","Amount"]]
            x1 = df1.drop(['Amount'],axis='columns')
            y1 = df1.Amount

        #create model

            df2=df1.reset_index()['Amount']
            from sklearn.preprocessing import MinMaxScaler
            scaler=MinMaxScaler(feature_range=(0,1))
            df3=scaler.fit_transform(np.array(df2).reshape(-1,1))
        ##splitting dataset into train and test split
            training_size=int(len(df3)*0.75)
            test_size=len(df3)-training_size
            train_data,test_data=df3[0:training_size,:],df3[training_size:len(df3),:1]

            def create_dataset(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]  
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)
        
            time_step = 5
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, ytest = create_dataset(test_data, time_step)
            X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        ### Create the Stacked LSTM model
            import tensorflow as tf
            #tf.config.experimental.list_physical_devices('GPU')
            # new_model=tf.keras.models.Sequential()
            # new_model.add(tf.keras.layers.LSTM(50,return_sequences=True,input_shape=(5,1)))
            # new_model.add(tf.keras.layers.LSTM(50,return_sequences=True))
            # new_model.add(tf.keras.layers.LSTM(50))
            # new_model.add(stf.keras.layers.Dense(1))
            # new_model.compile(loss='mean_squared_error',optimizer='adam')
            # new_model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=32,verbose=1)
            new_model = tf.keras.models.load_model('my_model.h5')
            new_model.summary()
            train_predict=new_model.predict(X_train)
            test_predict=new_model.predict(X_test)
            train_predict=scaler.inverse_transform(train_predict)
            test_predict=scaler.inverse_transform(test_predict)
            x_input=test_data[-5:].reshape(1,-1)
            x_input.shape
            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()

            from numpy import array

            lst_output=[]
            n_steps=5
            i=0
            while(i<30):
    
                if(len(temp_input)>5):
        #print(temp_input)
                    x_input=np.array(temp_input[1:])
                    print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
                    yhat =new_model.predict(x_input, verbose=0)
                    print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
        #print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat =new_model.predict(x_input, verbose=0)
                    print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1
    

            print(lst_output)
            day_new=np.arange(1,6)
            day_pred=np.arange(101,131)

            plt.plot(day_new,scaler.inverse_transform(df3[-5:]))
            plt.plot(day_pred,scaler.inverse_transform(lst_output))

            df4=df3.tolist()
            df4.extend(lst_output)
            df4=scaler.inverse_transform(df4).tolist()
            df5 = pd.DataFrame (df4)
            session["nown"] = df5.to_json()
            print(plt.plot(df4[-5:]))
            df4=scaler.inverse_transform(df4).tolist()
            print(plt.plot(df4))
            index=df1.index[-1]
            index=index+1
            prediction=df4[index]
            print(prediction)

            # per year prediction








            
            print(df)
            df_size= df.size
            df_info= df.isnull().sum().sum()
            if(df_info!=0):
                print("null")

            df_shape=df.shape
            df_head=df.head(10)
            df_column=list(df.columns)

            return render_template("user.html", df_table=df.head(),input=input,
                                      df_size=df_size,df_info=df_info,
                                      df_shape=df_shape,df_column=df_column,prediction=prediction,user=user)

            #return redirect(url_for("user",usr=user))
    else:
        return render_template("user.html", df_table=df.head(),input=input,user_list=user_list,
                                      df_size=df_size,df_info=df_info,
                                      df_shape=df_shape,df_column=df_column)

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"



if __name__=="__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)
    app.run(debug=True,use_reloader=True)