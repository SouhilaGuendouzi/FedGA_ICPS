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