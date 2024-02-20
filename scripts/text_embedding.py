from openai import BadRequestError, OpenAI
import os



#function to create text embeddings

os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()

def create_text_embeddings(df, positions ,cols):
    
    for (pos,col) in zip(positions, cols):

        for i in range(len(df)):
            print(i)
            response = client.embeddings.create(
            input= df.iloc[i,pos] ,
            model="text-embedding-ada-002")

            df.at[i,col] = response.data[0].embedding

def single_text_embedding(text):
    
    try:
        response = client.embeddings.create(
        input= text,
        model="text-embedding-ada-002")

        return response.data[0].embedding
    except BadRequestError as e:
        print(e)
        return None
        