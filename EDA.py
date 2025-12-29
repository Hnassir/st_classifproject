import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt



def app():

    st.title('📊 Exploratory Data Analysis')

    data_path=st.file_uploader('Upload a CSV file',type='csv')


    # to clear classif page once visiting eda second time
    if 'data_path' in st.session_state:
        st.session_state.clear()


    if data_path:

        st.write('uploaded')
        st.text('##############################################')

        data=pd.read_csv(data_path)

        # defining key of dict of session
        st.session_state['data_path']=data

        st.write('### Data Preview')
        st.write(data.head())

        st.write('### Descriptive Statistics')
        st.write(data.describe())

        st.write('### Correlation Matrix')
        if st.checkbox('Show Correlation Matrix'):
            matrix=data.corr(numeric_only=True)
            figure=px.imshow(matrix,text_auto=True,color_continuous_scale='viridis',width=800)
            st.plotly_chart(figure)

        st.write('### Advanced Visualizations')

        st.text('select columns for Visualixation')

        num_col=data.select_dtypes(include='number').columns
        if len(num_col)>=2:
            x_axis=st.selectbox('X-axis',options=num_col)
            y_axis=st.selectbox('Y-axis',options=num_col.drop(x_axis),index=None)

            #y_axis=st.selectbox('Y-axis',options=[col for col in num_col if col != x_axis],index=None)

            fig_scatter=px.scatter(data,x_axis,y_axis,title='scatter plot')
            st.plotly_chart(fig_scatter)
        else :
            st.warning('dataset has less than 2 numerical features')

        st.write('### Data Distribution')

        #target=st.pills('select target column',options=data.columns)

        feature=st.selectbox('Features',options=num_col[:-1])

        fig,axe=plt.subplots()
        axe.hist(data[feature],bins=20,edgecolor='black')
        axe.set_title(f'histogram of {feature}')
        axe.set_xlabel(feature)
        axe.set_ylabel('freauency')
        st.pyplot(fig)


if __name__=='__main__':
    app()