from lib2to3 import pytree
from urllib import response
import streamlit as st
import pytesseract
from PIL import Image
# from pdf2image import convert_from_path
import pandas as pd
import yake
import fitz
import nltk
from gtts import gTTS
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import os
import re

st.title("Extract info from Files")

st.sidebar.title('Hyper Params')

menu = ["Image","Dataset","DocumentFiles","About"]
choice = st.sidebar.selectbox("Select the type of data", menu)

no_of_keys = st.sidebar.slider('Select the no of keywords', 1, 20, 2, 2)

output = 'response'
output = st.selectbox('Select the type of output', ('keys', 'response'))

# pre processing the images
filters = ['Gaussian', 'Low pass', 'High Pass', 'System defined']
filter = st.sidebar.selectbox("Select the type of filter to preprocess the image", filters)

tes = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tes

extractor = yake.KeywordExtractor()
language = 'en'
max_ngram_size = st.sidebar.slider('Select the parameter for ngram', 1, 20, 3, 2)
deduplication_threshold = st.sidebar.slider('Select the parameter for DD threshold', 1, 10, 9, 1)
deduplication_threshold = deduplication_threshold/10
numOfKeywords = 100
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict= dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def rees(glo_text, keys):
    for key in keys[:no_of_keys]:
        # st.write(type(glo_text))
        sent_tokens = nltk.sent_tokenize(glo_text)
        word_tokens = nltk.word_tokenize(glo_text)
        sent_tokens.append(key)
        word_tokens = word_tokens + nltk.word_tokenize(key)
        TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        response = sent_tokens[idx]
        if(output == 'response'):
            st.write(' - ' + key + ':' + response)
        else:
            st.write(' - ' + key)
        response = re.sub("[^a-zA-Z0-9]","",response)
        myobj = gTTS(text=response, lang=language, slow=False)
        myobj.save("audio.mp3")  
        st.audio("audio.mp3", format='audio/ogg')
        os.remove("audio.mp3")

def load_image(image_file):
    img = Image.open(image_file)
    st.image(img, width=250)
    text = pytesseract.image_to_string(img)
    img.close()
    return text
    # text = pytesseract.image_to_string(img)

def load_pdf(data_file):
    doc = fitz.open(stream=data_file.read(), filetype="pdf")
    text = ""
    glo_text = ''
    for page in doc:
        text = text + page.get_text()
    glo_text += text
    keywords = custom_kw_extractor.extract_keywords(text)

    for kw in keywords[::-1]:
        if(kw[1] > 0.1):
            keys.append(kw[0])
    # st.write(keys)
    doc.close()
    return glo_text, keys

keys = []

def tes_image(image_file):
    if image_file != None:
        # add filters if time permits
        glo_text = ''
        # text = pytesseract.image_to_string(load_image(image_file)) # can add a specific language to detect the text on the screen
        # st.image(load_image(image_file),width=250)
        # st.write(text)
        text = load_image(image_file)
        glo_text += text
        keywords = custom_kw_extractor.extract_keywords(text)

        for kw in keywords[::-1]:
            if(kw[1] > 0.1):
                keys.append(kw[0])

        # st.write(keys)
        return glo_text, keys

def tes_doc(data_file):
    if data_file != None:
        tup = load_pdf(data_file)
        return tup

def convert_df_to_text(df):
    pass # implement key to text here using key2text package

if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file != None:
        file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
        st.write(file_details)
        glo_text, keys = tes_image(image_file)
        rees(glo_text, keys)

elif choice == "Dataset":
    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV",type=["csv"])
    if data_file != None:
        file_details = {"filename":data_file, "filetype":data_file.type, "filesize":data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        st.write(df)
        convert_df_to_text(df)


elif choice == "DocumentFiles":
    st.subheader("DocumentFiles")
    docx_file = st.file_uploader("Upload Document", type=["pdf","docx","txt"])
    if st.button("Process"):
        if docx_file is not None:
            file_details = {"filename":docx_file.name, "filetype":docx_file.type, "filesize":docx_file.size}
            st.write(file_details)
            glo_text, keys = tes_doc(docx_file)
            rees(glo_text, keys)


# extract using nltk to frame sentences