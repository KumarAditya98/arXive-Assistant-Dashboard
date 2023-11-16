#%%

import os
import io
import base64
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_text

#----------------------------------------------
# dependencies

# pip install unstructured
# pip install "unstructured[all-docs]"
# pip install pdf2image
# pip install pdfminer
# pip install pdfminer-six
# pip install unstructured_pytesseract

# For mac
# brew install poppler

# For ubuntu
# sudo apt-get install poppler-utils
# sudo apt install tesseract-ocr

#----------------------------------------------
# An example
# with open("rpaper_1.pdf", mode='rb') as f:
#     binary = f.read()

# binary = io.BytesIO(binary)
# elem = partition_pdf(file = binary)
# text = convert_to_text(elem)        

#----------------------------------------------
# display pdf
def displayPDF(upload_values):
        
    base64_pdf = base64.b64encode(upload_values).decode('utf-8') # fixes seek problem

    pdf_display = pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    return pdf_display

#----------------------------------------------
# data preprocessing

def preprocessing(upload_values):
    
    upload_values = io.BytesIO(upload_values) # fixes seek problem
    elem = partition_pdf(file = upload_values)
    text = convert_to_text(elem) 
    return text


# %%
