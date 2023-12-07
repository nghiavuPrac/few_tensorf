import streamlit as st
import numpy as np
import os
import glob
from streamlit_page.train_model.stl_data_preparation import *
from streamlit_page.train_model.stl_data_visualization import *
from streamlit_page.train_model.stl_obj_visualization import *
from streamlit_page.train_model.stl_training import *
def train_model():
    st.header('Data preparation')
    
    prepare_data, training, vis_image, vis_obj_3d = st.tabs(["Create config", "Training", "Visualize image", 'Visualize 3d object'])
    
    data_folder = r'data\data'
    config_folder = r'few_nerf\configs'
    obj_folder = r'data\object_data'

    with prepare_data:
        data_preparation(data_folder, config_folder)
    
    with training:
        training_model(config_folder)

    with vis_image:
        data_visualization(data_folder)
    
    with vis_obj_3d:
        obj_visualization(obj_folder)