# Simplifying AI Research Papers 
## – A One-Stop-Shop for Accessing and Understanding AI Literature

## Background:

AI Research papers are available in abundance in today’s time, with cutting edge development happening almost every day. One of the most popular destinations to access these scholarly articles and research papers is the arXiv platform which is an open-source archive for research papers in various fields. 

The problem arises when we want to access articles pertaining to specific topics or a combination of topics, it is difficult to find the right resource without having to resort to a google search. Additionally, not everyone is equipped with the time, effort, and/or capability to fully dive into the details of the technical aspects of the research paper they are referencing and find answers to the exact question in mind. 

## Proposal:

This is where we decided to leverage the power of NLP to develop an innovative solution to this problem. We envision a comprehensive web application for everything related to accessing and analyzing AI related scholarly articles. The dataset that we have picked up is of 10,000 scholarly research papers and articles related to AI from the arXiv website, for access to the dataset see below. Some of the functionalities of the design solution will be as follows – 
1.	A page in the UI where Users can upload the research paper and/or any article they want to analyze in depth through our platform. On this uploaded research paper, we will provide the following functionalities:
a.	A comprehensive summarization of the entire document. 
b.	A Quention and Answering chatbot to answer specific questions about the uploaded research paper and/or document. 
2.	Additional functionalities are as follows – 
a.	Users will be able to utilize the chatbot to retrieve research papers on a specific topic or a combination of topics. For this we wish to utilize the corpus we will be working with. Zero-shot classification will be explored here. 
b.	We will attempt to perform multi-label classification for all the research papers that we already have, to incorporate with the above feature. 
c.	Perform some text analytics on the uploaded document and visualize this for the Users. for a specific research paper retrieval based on provided topics. 

The libraries and packages that we hope to work with are as follows – LangChain, llama-index, transformers, PyTorch, etc. 

## Repository & Dataset Source
Dataset Source: Click HERE to view the source of our dataset.
GitHub Repo: Follow this LINK to view our GitHub repository.
