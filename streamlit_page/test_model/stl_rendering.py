import streamlit as st
import os
import glob
from few_nerf.opt import *

def rendering(log_dir):
    object_option = st.selectbox(
        "Select object", 
        os.listdir(log_dir), 
        key='object_option', 
        index=None,
    )

    data_path = 'data/data'
    dataset_type = st.selectbox(
        "Select data Type", 
        os.listdir(data_path), 
        key='dataset_type', 
        index=None,
    )

    datadir = ''
    if dataset_type:
        dataset_folder = st.selectbox(
            "Select data folder", 
            os.listdir(os.path.join(data_path, dataset_type)), 
            key='dataset_folder', 
            index=None,
        )   
        if dataset_folder:
            datadir = os.path.join(data_path, dataset_type, dataset_folder)

            render_test_box = st.selectbox(
                "Choice render test", 
                [True, False], 
                key='render_test_box', 
            )

            render_train_box = st.selectbox(
                "Choice render train", 
                [True, False], 
                key='render_train_box'
            )

            if object_option:
                cmd_arguments = [
                    '--config',
                    os.path.join(log_dir, object_option, 'final_'+object_option+'.txt'),
                    '--ckpt',
                    os.path.join(log_dir, object_option, 'final_'+object_option+'.th'),
                    '--datadir',
                    datadir,
                    '--render_only',
                    '1',
                    '--render_test',
                    '1' if render_test_box else '0',
                    '--render_train',
                    '1' if render_train_box else '0'
                ]

                args = config_parser(cmd_arguments)
                
                render_button = st.button(
                    "Show back pic ⏭️",
                    key = 'render_button'
                )

                if render_button:
                    render_test(args)

                    # images_path = os.path.join(data_dir, dataset_option, dataset_obj_option, dataset_type_option)
                    images_path = r'few_nerf\log\raw_8_0300\imgs_test_all'
                    image_list = [os.path.join(images_path, iamge_name) for iamge_name in  os.listdir(images_path) if 'png' in iamge_name]
                                        
                    col1,col2 = st.columns(2)

                    if 'counter' not in st.session_state: 
                        st.session_state.counter = 0

                    def showPhoto(next):
                        col1.write(f"Index as a session_state attribute: {st.session_state.counter}")
                        
                        ## Increments the counter to get next photo
                        if next:
                            st.session_state.counter += 1
                            if st.session_state.counter >= len(image_list):
                                st.session_state.counter = 0
                        else:
                            st.session_state.counter -= 1
                            if st.session_state.counter < 0:
                                st.session_state.counter = len(image_list)-1

                        # Select photo a send it to button
                        photo = image_list[st.session_state.counter]
                        col2.image(photo,caption=photo)

                    # Get list of images in folder
                    col1.subheader("List of images in folder")
                    col1.write(image_list)


                    with col1:
                        bt_col1, bt_col2 = st.columns(2)
                        show_back_btn = bt_col1.button(
                            "Show back pic ⏭️",
                            on_click=showPhoto,
                            args=([False]),
                            key = 'show_back_btn'
                        )
                        show_next_btn = bt_col2.button(
                            "Show next pic ⏭️",
                            on_click=showPhoto,
                            args=([True]),
                            key = 'show_next_btn'
                        )
