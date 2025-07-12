
import streamlit as st


def home_page() -> None:
        st.set_page_config(page_title="APCAS 2.0 | Home", page_icon='🤖')


        # ----------------------------------------------------------- Sidebar --------------------------------------------------------------- #
        st.sidebar.header('Menu', divider=True)
        st.sidebar.write('It is basically a RAG (Retrieval Augmented Generation) based web application, \
                        that optimized for chatting with any PDF in an efficint way')
        if st.sidebar.button(label='Chat Now', key='chat_now_sidebar_btn_key', type='primary'):
            st.session_state.page = 'working'

        st.sidebar.subheader("Connet with me!")
        st.sidebar.write("[Kaggle](https://www.kaggle.com/architty108)")
        st.sidebar.write("[Github](https://www.github.com/a4archit)")
        st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/a4archit)")





        # ---------------------------------------------------------- Body -------------------------------------------------------------------- #
        st.header('APCAS 2.0', divider=True)

        st.write('APCAS stands for Any PDF Chatting AI System, it is basically a RAG (Retrieval Augmented Generation) based \
                application, that optimized for chatting with any PDF in an efficint way, still it is not a professional  \
                project So it may have several bugs.')
        
        st.markdown('''### Technologies used in this Project
                    
- Langchain
- Version Control (Git & Github)
- Web Application (Using Streamlit)
- OOPs (Object Oriented Programming System)
- Python with its libraries and some more techs
        ''')

        if st.button(label='Chat Now', key='chat_now_btn_key', type='primary'):
              st.session_state.page = 'working'



        st.subheader('Model Working Flow', divider=True)

        st.image('./materials/dummy_image.png')


        



if __name__ == "__main__":
      
      home_page()