import streamlit as st
import numpy as np
import os
import glob
from streamlit_page.train_model.stl_data_preparation import *
from streamlit_page.train_model.stl_data_visualization import *
from streamlit_page.train_model.stl_obj_visualization import *
def train_model():
    st.header('Data preparation')
    
    prepare_data, vis_image, vis_obj_3d = st.tabs(["Create config", "Visualize image", 'Visualize 3d object'])
    
    with prepare_data:
        config_folder = r'few_nerf\configs'
        data_preparation(config_folder)

    with vis_image:
        data_folder = r'data\data'
        data_visualization(data_folder)
    
    with vis_obj_3d:
        obj_folder = r'data\object_data'
        obj_visualization(obj_folder)