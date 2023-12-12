import streamlit as st
import requests
from bs4 import BeautifulSoup
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Load tokenizer and model
tokenizer = LEDTokenizer.from_pretrained("bakhitovd/led-base-7168-ml")
model = LEDForConditionalGeneration.from_pretrained("bakhitovd/led-base-7168-ml").to("cuda")

import re

# def get_all_links(content):
#     soup = BeautifulSoup(content, 'html.parser')
#     links = []
#     for a in soup.find_all('a', href=True):
#         href = a['href']
#         # Filter out links that are likely to be irrelevant
#         if href.startswith('http') and not any(x in href for x in ['@', '?source=', '/tag/', 'help.medium.com', 'medium.statuspage.io', 'policy.medium.com', 'speechify.com']):
#             links.append(href)
#     return links

def get_all_links(content):
    soup = BeautifulSoup(content, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Filter to include only clean-looking links
        if href.startswith('http') and '?' not in href and not href.endswith('/'):
            links.add(href)
    return links


def process_text(summary_list):
    # Convert the list to a string
    summary_text = ' '.join(summary_list)

    # Remove single quotes at the start and end, if present
    summary_text = summary_text.strip("'")

    # Capitalize the first letter of each sentence
    summary_text = '. '.join(sentence.capitalize() for sentence in summary_text.split('. '))
    summary_text =  summary_text + "."

    return summary_text

# Function to summarize the article
def summarize(the_article):
    inputs_dict = tokenizer(the_article, padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, max_length=512)
    return tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)

def get_article(url):
    response = requests.get(url)
    return response.content

# Function to get the article from URL
def get_article_text(url):
    response = requests.get(url)
    content = BeautifulSoup(response.content, 'html.parser')
    title = 'The article'

    try:
        title = content.find('title').text
    except:
        print('There is no title')

    text = content.find_all('p')
    article = ''
    for p in text:
        if len(p.text) > 100 and p.text[0] != '[':
            article = article + ' ' + p.text
    return title, article

# Streamlit interface
st.title("MediumBlink: The Rapid Insight & Web Weaver ")

# Input for URL
url = st.text_input("Enter the URL of the article:")

if url:
    title, content1 = get_article_text(url)
    content2 = get_article(url)
    summary = summarize(content1)
    links = get_all_links(content2)
    processed_text = process_text(summary)

    st.subheader("Summary")
    st.markdown(processed_text)
    st.markdown(f"[Read the full article here]({url})", unsafe_allow_html=True)

    st.subheader("Links in the Article")
    for link in links:
        st.markdown(f"[{link}]({link})", unsafe_allow_html=True)

# url = "https://medium.com/@wesleywarbington_22315/ai-stock-trading-d71955621834"
