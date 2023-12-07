import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pyvista as pv
from stpyvista import stpyvista
sys.path.insert(5, './')
from stl_dataset import dataset_mode
from streamlit_page.train_model.stl_train import *
from streamlit_page.test_model.stl_test import *
# Side bar
with st.sidebar:
    st.markdown('# **3D reconstruction**')
    option = st.selectbox(
        'Select option',
        ['Train model', 'Test model', 'Model'],
        key='select_option'
    )


# Option
if option == 'Train model':
    train_model()
elif option == 'Test model':
    test_model()
else:
    st.header("Model")
    train_tab, valuation_tab, inference_tab = st.tabs(["Train", "Valuation", "inference"])
    with train_tab:
        pass
    with valuation_tab:
        pass
    