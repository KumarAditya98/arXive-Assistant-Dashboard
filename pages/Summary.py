#%%
import streamlit as st
from Utils import *
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#----------------------------------------------
# Dependencies

# pip install sentencepiece

#----------------------------------------------
# Dummy Model

def summarize(text):
  return summarizer(text, max_length=150, min_length=1, do_sample=False)

#----------------------------------------------
# Main function
def main():
    upload = st.file_uploader("Upload an Arxiv Paper:", type=["pdf"])
    
    if upload is None:
        st.stop()
        
    pdf_display = displayPDF(upload_values=upload.getvalue())
    
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    input = st.text_input(label="Paste for summary here! :smile:")
    
    summary = summarize(input)
    
    st.write(summary)
    
if __name__ == "__main__":
    main()
