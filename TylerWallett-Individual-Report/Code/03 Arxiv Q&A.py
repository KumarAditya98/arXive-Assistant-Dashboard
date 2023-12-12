#%%

# Credit to: https://www.youtube.com/watch?v=9SBUStfCtmk
# Credits to: https://www.youtube.com/watch?app=desktop&v=9ISVjh8mdlA

import streamlit as st
import pandas as pd
from Chains import construct_chain, wrap_text_preserve_newlines

# Title
st.title("Arxiv Q&A")

st.snow()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("my_form"):
    st.write("Topic of interest:")
    topic = st.text_input("", key="topic")
    st.write("Question:")
    prompt = st.text_input("", key ="prompt")
    
    # Every form must have a submit button.
    submit = st.form_submit_button("Submit")

if submit != False:
    chain = construct_chain(topic = topic)

    llm_response = chain(prompt)

    response = wrap_text_preserve_newlines(llm_response['result'])

    metadata = [[source.metadata['title'], source.metadata['file_path']] for source in llm_response["source_documents"]]

    df = pd.DataFrame(metadata, columns = ["Titles", "URL"])

    with st.chat_message("assistant"):
        st.markdown(response)
        st.dataframe(df,
                    column_config = {"URL": st.column_config.LinkColumn("URL")},
                    hide_index = True)
    st.session_state.messages.append({"role": "assistant", "content": response}) 


