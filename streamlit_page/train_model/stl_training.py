import streamlit as st
import os
from few_nerf.opt import *
from few_nerf.train import reconstruction


def training_model(config_dir):
    
    config_option = st.selectbox(
        "Select config", 
        os.listdir(config_dir), 
        key='config_option', 
        index=None, 
    )

    if config_option != None:
        cmd_arguments = [
            '--config',
            os.path.join(config_dir, config_option)
        ]

        args = config_parser(cmd_arguments)



        train_button = st.button(
            'Start training',
            key = 'train_button'
        )        

        if train_button:
            stop_button = st.button(
                'Stop training',
                key = 'stop_button'
            )        
            with st.spinner('Wait for it...'):
                reconstruction(args)                
