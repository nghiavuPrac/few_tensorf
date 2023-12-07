import streamlit as st
import numpy as np
import os
import glob
import configargparse
from few_nerf.dataLoader.__init__ import *
from few_nerf.opt import *
from streamlit_page.test_model.stl_rendering import *
from streamlit_page.test_model.stl_mesh_extract import *

def test_model():
    st.header('Testing model')

    rendering_tab, mesh_extract_tab = st.tabs(["Rendering", "Mesh extract"])

    log_dir = 'few_nerf\log'
    with rendering_tab:
        rendering(log_dir)
    
    with mesh_extract_tab:
        mesh_extract(log_dir)
            

