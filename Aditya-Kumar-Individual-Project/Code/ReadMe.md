# Code Description
Run the requirements.txt to install all external dependencies using command: pip install -r /requirements.txt

## Code Files Included in Analysis
* The Code File named Document-Upload-Q&A-Streamlit-V2.py contains the document upload Q&A utility page. This python file has been fetched into Tyler Wallet's Code section to be used as the page for XAI product offering. An important point to note regarding this project is that it utilizes Hugging Face API token that can be accessed through this [link](https://huggingface.co/settings/tokens). There are two ways that this can be achieved.      
   1. Either open the python file and hard-code your Hugging Face API Token at the required position.
   2. Run these commands on your Unix/Linux Terminal:   
        cd ~    
        echo 'export HF_HOME_TOKEN="<your-hugging-face-token>"' >> ~/.bashrc    
        source ~/.bashrc    
  Upon doing either of these steps, the python file should produce desired result.
* The arXiv-TopicModelling-Streamlit.py contains the interactive Topic Modelling page. This python file has been fetched into Tyler Wallet's Code section to be used as the page for XAI product offering.

## Extra Code Files
* Python files beginning with "Testing-" were used to get familiar with concepts and test out certain utilities. These files may not run as it is and can only be used as a reference. 
