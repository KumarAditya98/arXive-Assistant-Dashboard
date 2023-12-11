from IPython.display import HTML, display
import requests
from bs4 import BeautifulSoup
import torch

from transformers import LEDTokenizer, LEDForConditionalGeneration

tokenizer = LEDTokenizer.from_pretrained("bakhitovd/led-base-7168-ml")

model = LEDForConditionalGeneration.from_pretrained("bakhitovd/led-base-7168-ml").to("cuda")

def display_html(text, header):
    text_to_display =  f"""
    <html>
      <head>
        <title>{header}</title>
      </head>
      <body>
        <h2> {header} </h2>
        <p>{text}</p>
      </body>
    </html>
    """
    display(HTML(text_to_display))

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

url = 'https://towardsdatascience.com/using-transformers-for-computer-vision-6f764c5a078b'

#url = 'https://medium.com/@wesleywarbington_22315/ai-stock-trading-d71955621834'

#url = 'https://medium.com/artificial-corner/bye-bye-chatgpt-ai-tools-better-than-chatgpt-but-few-people-are-using-them-eac93a3627cc'

title, article = get_article(url)

summary = summarize(article)

display_html(summary,'Summary')

display_html(article, title)
