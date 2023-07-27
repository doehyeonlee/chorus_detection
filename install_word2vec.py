import os
import requests
from gensim.models import KeyedVectors

word2vec_model_path = 'saved_word2vec_model.model'
download_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip'

if os.path.exists(word2vec_model_path):
    word2vec_model = KeyedVectors.load(word2vec_model_path)
else:
    # Download the model
    response = requests.get(download_url)
    zip_path = 'wiki-news-300d-1M-subword.vec.zip'
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # Extract the ZIP
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

    # Load the model into memory
    word2vec_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subword.vec", binary=False)

    # Save the model locally
    word2vec_model.save(word2vec_model_path)

    # Optional: Clean up by removing downloaded files
    os.remove(zip_path)
    os.remove("wiki-news-300d-1M-subword.vec")
