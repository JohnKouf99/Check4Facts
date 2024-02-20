from requests.exceptions import SSLError
from bs4 import BeautifulSoup, Comment
import requests
import sys
sys.path.append('scripts')  
from text_embedding import single_text_embedding
import numpy as np
import pandas as pd
import datetime
from scipy.spatial.distance import cosine
from langdetect import detect
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re
import time
import signal






class Harvester:


    def __init__(self, url_list, claim, timeout, claim_id):
        self.timeout = timeout
        self.url_list = url_list
        self.claim = claim
        self.claim_id = claim_id
        self.timeout_seconds = 20 * 60
        


    def signal_handler(signum, frame):
        raise TimeoutError("Function execution timed out")


    

    
    def get_html_text(self, url):
        
           
        try:
            response =  requests.get(url, timeout=self.timeout, verify=False)
            time.sleep(2)
            if response.status_code == 200:
                    return BeautifulSoup(response.text, 'html.parser')
            else: 
                return None
        except SSLError as e:
            print("SSL Error:", e)
            print('On url: ', url)
            return None
        except Exception as e:
        
            print("An error occurred:", e)
            print('On url: ', url)
            return None

      
           


    def get_title(self,soup):
        
        title = soup.find(lambda tag:"title" in tag.get('class', []))
        if title:
            return title.text.strip()
        elif soup.find('meta', property='og:title'):
            title = soup.find(
                'meta', property='og:title').get('content').strip()
        elif soup.find('h1'):
            title = soup.find('h1').text.strip()
        else: 
            return None
        return title
            


    # def text_preprocess(self, texts):
            
    #     filtered_arr = [text for text in texts if len(text) > 1 and len(text.split()) > 3]
    #     return " ".join(filtered_arr)

    def get_body(self, soup):
    
        [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title', 'footer'])]
    
        body_text =  soup.get_text(separator ='\n', strip=True)

        texts = body_text.split('\n')
        
        result = "\n".join(text.strip() for text in texts if len(text.split())>3)  #.replace("\n", " ")

        return result

    def similary_text(self,claim, texts):
        claim_emb = single_text_embedding(claim)

        # spaw to texts se paragrafous kai vriskw most similar paragraph
        #for each paragraph spaw se protaseis? kai vriskw similar sentence

        paragraphs = texts.split('\n')
        sentences = sent_tokenize(texts)

        # for s in sentences:
        #     print(s)

        # for text in texts:
        #     print(text)
            
        #paragraph_tuples
        tuples  = [( np.dot(claim_emb, single_text_embedding(text)), text) for text in paragraphs if not self.is_english(text) 
                   and len(text.split())>3 and single_text_embedding(text) is not None]
        if not tuples:
            print('paragraph tuples is empty')
            print(texts)
            print('---------------')
            print(paragraphs)
        # #sentence_tuples
        tuples2  = [( np.dot(claim_emb, single_text_embedding(text)), text) for text in sentences if not self.is_english(text) 
                    and len(text.split())>3 and single_text_embedding(text) is not None]
        if not tuples2:
            print('sentence tuples is empty')


        # #exw tis paragraphous kai tis spaw se protaseis me tin nltk 
        # for p in paragraphs:
        #     print(p)
        #tuples2  = [( np.dot(claim_emb, single_text_embedding(p)), text) for s in paragraph if not self.is_english(p)]
        if tuples:
            similarity , result = max(tuples, key=lambda x: x[0])
        else:
            similarity, result = None, None
        if tuples2:
            similarity2 , result2 = max(tuples2, key=lambda x: x[0])
        else: 
            similarity2, result2 = None,None

        return similarity, result, similarity2, result2




    def run(self):

        signal.signal(signal.SIGALRM, self.signal_handler)


        df = pd.DataFrame(columns=['id','claim_id', 'title', 'body', 'most_similar_sentence', 'most_similar_paragraph', 
                                   'harvest_date', 'url', 'most_similar_par_cos','most_similar_sent_cos'])

        for url in self.url_list: #na valw orio claims
            html = self.get_html_text(url)
            if html is None:
                print(f'''Invalid url: {url}, 
                      skipping procedure....''')
                continue
            title = self.get_title(html)
            body = self.get_body(html)

            if((len(body.split())<50) or body is None):
                print('body is none')
                continue
            print(f'''
            URL: {url}
            Title: {title}
            ''')

            try:
                signal.alarm(self.timeout_seconds)
                similarity_p, result_p, similarity_s, result_s = self.similary_text(self.claim, body)
                signal.alarm(0)
            except TimeoutError:
                print("Function execution timed out. Skipping function and continuing with the rest of the code.")
                continue

            data = {'id': len(df),'claim_id': self.claim_id, 'title': title, 'body': body.replace("\n", " "), 
                    'most_similar_sentence': result_s.replace('\n',' ').replace('\xa0',''), 'most_similar_paragraph': result_p.replace('\xa0',''), 
                    'harvest_date': datetime.date.today() , 'url': url, 'most_similar_par_cos': result_p, 'most_similar_sent_cos':result_s}

            df.loc[len(df)] = data

            if(len(df)>=5):
                break

            
            print()
            print(f'''Most similar paragraph: {result_p}
Cosine similarity: {similarity_p}
Most similar sentence: {result_s}
Cosine similarity: {similarity_s}
                  ''')


        return df
            


    def is_english(self, text):
        try:
            return detect(text) == 'en'
        except:
            # Handle cases where language detection fails
            return False
        
