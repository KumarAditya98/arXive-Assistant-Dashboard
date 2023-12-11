# pip install arxiv
import arxiv  # Interact with arXiv api to scrape papers
from sentence_transformers import (
    SentenceTransformer,)  # Use Hugging Face Embedding for Topic Modelling
from bertopic import BERTopic  # Package for Topic Modelling
from tqdm import tqdm  # Progress Bar When Iterating
import glob  # Identify Files in Directory
import os  # Delete Files in Directory
from unstructured.partition.auto import partition  # Base Function to Partition PDF
from unstructured.staging.base import (
    convert_to_dict,
)  # Convert List Unstructured Elements Into List of Dicts for Easy Parsing
from unstructured.cleaners.core import (
    clean,
    remove_punctuation,
    clean_non_ascii_chars,
)  # Cleaning Functions
import re  # Create Custom Cleaning Function
import nltk  # Toolkit for more advanced pre-processing
from nltk.corpus import stopwords  # list of stopwords to remove
from typing import List  # Type Hinting
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


nltk.download("stopwords")

def get_arxiv_paper_texts(query, max_results = 100):

    arxiv_papers = list(
        arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        ).results()
    )

    paper_texts = []
    for paper in tqdm(arxiv_papers):
        paper.download_pdf()
        pdf_file = glob.glob("*.pdf")[0]
        elements = partition(pdf_file)
        isd = convert_to_dict(elements)
        narrative_texts = [
            element["text"] for element in isd if element["type"] == "NarrativeText"
        ]
        os.remove(pdf_file)
        paper_texts += narrative_texts
    return paper_texts


# Function to Apply Whatever Cleaning Functionality to Each Narrative Text Element
def custom_clean_function(narrative_text: str) -> str:
    stop_words = set(stopwords.words("english"))
    remove_numbers = lambda text: re.sub(
        r"\d+", "", text
    )
    cleaned_text = remove_numbers(narrative_text)
    cleaned_text = clean(
        cleaned_text,
        extra_whitespace=True,
        dashes=True,
        bullets=True,
        trailing_punctuation=True,
        lowercase=True,
    )
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = " ".join(
        [word for word in cleaned_text.split() if word not in stop_words]
    )
    return cleaned_text

def get_intertopic_dist_map(topic_model):
    return topic_model.visualize_topics()

def get_topic_keyword_barcharts(topic_model):
    return topic_model.visualize_barchart(top_n_topics=9, n_words=5, height=400)

def main():
    paper_texts = False
    st.title("Topic Modeling with arXiv data.")
    input = st.text_input("Input any topic that comes to mind.")
    if input:
        data_load_state = st.text('Loading data...')
        paper_texts = get_arxiv_paper_texts(query=input, max_results=10)
        data_load_state.text('Fetching relevant articles... done!')

    elif st.button('Load demo topic: Natural Language Processing'):
        data_load_state = st.text('Loading data...')
        paper_texts = get_arxiv_paper_texts(query="natural language processing", max_results=10)
        data_load_state.text('Fetching relevant articles... done!')

    if paper_texts:
        data_clean_state = st.text('Cleaning data...')
        cleaned_paper_texts = [custom_clean_function(text) for text in paper_texts]
        data_clean_state.text('Cleaning data... done!')
        tm_state = st.text('Modeling topics...')
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic(embedding_model=sentence_model, verbose=False, min_topic_size=50)
        topics, _ = topic_model.fit_transform(cleaned_paper_texts)
        tm_state.text('Modeling topics... done!')
        st.markdown("""---""")

        freq = topic_model.get_topic_info();
        st.write(freq.head(10))

        fig3 = get_topic_keyword_barcharts(topic_model)
        st.plotly_chart(fig3)

        fig1 = get_intertopic_dist_map(topic_model)
        st.plotly_chart(fig1)

        fig2 = topic_model.visualize_heatmap(n_clusters=5, top_n_topics=10)
        st.plotly_chart(fig2)

if __name__ == '__main__':
    main()