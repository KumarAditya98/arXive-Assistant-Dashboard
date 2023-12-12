#%%
import streamlit as st

# To run the page: streamlit run About.py --server.port 8888

#----------------------------------------------
# Main function

def main():
    st.title("One stop shop for learning")
    st.image("question.jpg")
    st.divider()
    st.subheader("NLP Team 2")
    
    """
    * Arxiv topic modeling
    * Document upload
    * Arxiv Q&A
    * MediumBlink
    """
    
    st.caption("By Medhasweta Sen, Aditya Kumar and Tyler Wallett")
    
if __name__ == "__main__":
    main()

#%%
