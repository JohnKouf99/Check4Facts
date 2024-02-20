from textblob import TextBlob
from textblob.exceptions import TranslatorError




def translate(text_to_translate):
    
    blob = TextBlob(text_to_translate)
    
    # #Translate the text to another language 
    # translated_text = translator.translate(text_to_translate, dest='el')
    try:
        return blob.translate(from_lang='en', to='el')
    except TranslatorError as e:
    # Handle NotTranslated exception
        print(text_to_translate)
        print(f"Translation not possible: {e}")


