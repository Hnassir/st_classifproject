import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def app () :

    st.title('✅Classification')


    # verifying session key value if EDA was checked
    if 'data_path' in st.session_state:
        df=st.session_state['data_path']



        st.write('data loaded for classification')

        #st.write(df)
        st.dataframe(df.head())

        target_col=df.columns[-1]


        # we use the function for html and css code

        st.markdown(f""" 
            <div style="background-color: rgb(249, 249, 249); border-radius: 10px; padding: 10px; text-align: center; margin-bottom: 20px;"> 
                <h3 style="color: rgb(0, 120, 212)"> Target Column: <span style="font-weight: bold;">{target_col}</span> </h3>
            </div>
            """, unsafe_allow_html=True)

        x=df.drop(columns=target_col)
        y=df[target_col]


        size=st.slider('test size (as %)',min_value=10,max_value=35,value=20)/100
        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=size, random_state=42)


        model=st.selectbox('choose model',options=['logistic reg','decision tree','random forest','KNN'])

        if model =='logistic reg' :
            clf=LogisticRegression()
        elif model =='decision tree' :

            criterion=st.radio('type of criterion',options=['gini','entropy'])
            depth=st.slider('max depth',min_value=5,max_value=20,value=5)
            sample_leaf=st.slider('min sample leaf',min_value=10,max_value=50,value=20)

            clf=DecisionTreeClassifier(criterion=criterion,max_depth=depth,min_samples_leaf=sample_leaf)

        elif model =='random forest' :

            n_estim=st.slider('number of estimators',min_value=10,max_value=100,value=20,step=5)
            depth=st.slider('max depth',min_value=2,max_value=20,value=10)

            clf=RandomForestClassifier(n_estimators=n_estim,max_depth=depth)

        else :

            k=st.slider('number of neighbors',min_value=2,max_value=20,value=5)
            metric=st.selectbox('metric',options=['minkowski','manhattan','euclidean'])
            
            p=2
            if metric=='minkoski':
                p=st.slider('power parameter for the minkowski metric',min_value=3,max_value=5,value=3)

            clf =KNeighborsClassifier(n_neighbors=k,metric=metric,p=p)


        if st.button('train model'):

            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)

            tab1,tab2=st.tabs(['metrics','confusion matrix'])

            with tab1:
                st.write('### performence metrics')
                #report=classification_report(y_test,y_pred)
                #st.write(report)
                report=classification_report(y_test,y_pred,output_dict=True)
                #st.write(report)
                #report_df=pd.dataframe(report)
                #st.dataframe(report_df)

                report_df=pd.DataFrame(report).transpose()
                #st.write(report_df)
                st.dataframe(report_df.style.format(precision=2))


            with tab2:
                st.write('### confusion matrix')
                conf=confusion_matrix(y_test,y_pred)

                display=ConfusionMatrixDisplay(conf,display_labels=clf.classes_)

                fig,axe=plt.subplots()
                display.plot(ax=axe,cmap='viridis',colorbar=False)
                st.pyplot(fig)
    
    else :

        st.write('No data found. Please upload a file on the EDA page first.')