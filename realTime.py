import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import time
import numpy as np
import random

st.write(
    """
# Biopsias classification demo
"""
)

print(st.session_state)
a='y'
text_area = st.empty()

text = text_area.text_area("Text to analyze", "Write a number")

button1 = st.button("Random")
button2 = st.button("Run")

if button1:
    a=a+str(random.random())
    text = text_area.text_area("Text to analyze", a)

if button2:
    st.write(text)

    ##############################################



     def ui(self, t):
   
       if 'edge' not in st.session_state:	st.session_state.edge = edge

    #t=threading.Thread(target=edge.ui, args=("hi", ))
       get_script_run_ctx()
    #t.start()
      
       if 'text_area' not in st.session_state:
           st.session_state.text_area = " "
      
       st.markdown(f"# Edge {self.id} ")
       image = Image.open('pictures/logoa.png')
      
       st.sidebar.image(image)
       st.sidebar.markdown(f"# Edge {st.session_state.edge.id}")

    
       st.sidebar.write(f"Ip Address: {HOST}")
       st.sidebar.write(f"Domain: {st.session_state.edge.domain}")
       st.sidebar.write(f"Task: {edge.task}")
       st.sidebar.write(f"Train Accuracy: {st.session_state.edge.accuracy[0]}")
       st.sidebar.write(f"Test Accuracy: {st.session_state.edge.accuracy[1]}")
    
       col1, col2, col3, col4 = st.sidebar.columns(4)
       with col1:
        connect=st.sidebar.button(label='Connect',key='connect')
       with col2:
         train=st.sidebar.button(label='Train',key='Train')
       with col3:
        upload = st.sidebar.button(label="Upload",key="Upload")
       with col4:
         request=st.sidebar.button(label="Request for Model",key="Request for Model")
    #with col2:
       if connect:
            #st.session_state.text_area="souhila"
            st.session_state.edge.connect()
         
       if train:
          #print(st.session_state.text_area)
          
          st.session_state.edge.Training(False)

        
       elif upload :
            st.session_state.edge.send_message(edge.model.state_dict(),'LocalModel')

     
       if request:
             st.session_state.edge.TransferLearningRequest()
         
    
    

       st.text_area('Output', value=st.session_state.edge.text,height=500, disabled=True)
     #st.write(st.session_state.edge.text)
       chart_data = pd.DataFrame(
     np.random.randn(20, 3),
      columns=['a', 'b', 'c'])

       st.line_chart(chart_data, width=500,height=500) 