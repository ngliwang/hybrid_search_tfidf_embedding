import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gc
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import faiss
import gc
import time
from sklearn.metrics.pairwise import linear_kernel
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# ================= LOAD DATA ==================================
# load wiki_movie_plots_deduped.csv top 500 rows into a DataFrame
data = pd.read_csv('wiki_movie_plots_deduped.csv', nrows=500)
df = data[['Title','Plot']]
del data
gc.collect()
# df.dropna(inplace=True)
# df.drop_duplicates(subset=['Plot'],inplace=True)
print('done loading data')



# ================= PREPROCESSING ===============================
# Initialize the Lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert the text to lower case
    text = text.lower()

    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words and punctuation, and perform lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stopwords.words('english')]

    return ' '.join(words)

# Apply the preprocess_text function to the 'Plot' column
df['processed_plot'] = df['Plot'].apply(preprocess_text)
print('done preprocessing')



# ================== TF-IDF =====================================
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# if tfidf_matrix.pickle exists, load it
if os.path.exists('tfidf_matrix.pickle'):
    with open('tfidf_matrix.pickle', 'b') as handle:
        tfidf_matrix = pickle.load(handle)
    print('done loading tfidf_matrix.pickle')
else:    
    # Compute the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(df['processed_plot'])

# save tfidf_matrix as a pickle file
with open('tfidf_matrix.pickle', 'wb') as handle:
    pickle.dump(tfidf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(tfidf_matrix)
print("done tf-idf")



# ===================== embedding based ==================
model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3', cache_folder="./cache")

# USE CASE FOR 500!!!!
encoded_data = np.load(os.path.join(os.path.dirname(__file__), 'encoded_data_500.npy'))

# # create the index
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(df))))
# faiss.write_index(index, 'movie_plot.index')

# load movie_plot.index as index
# index = faiss.read_index('movie_plot.index')


def fetch_movie_info(dataframe_idx):
    info = df.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['Title'] = info['Title']
    meta_dict['Plot'] = info['Plot']  
    return meta_dict

# ===================== search function ==================
def search_tfidf(query):
    query = preprocess_text(query)
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_tfidf, tfidf_matrix).flatten()
    normalized_cosine_similarities = (cosine_similarities - min(cosine_similarities)) / (max(cosine_similarities) - min(cosine_similarities))
    top_10 = normalized_cosine_similarities.argsort()[:-11:-1]
    return df[['Title', 'Plot']].iloc[top_10], normalized_cosine_similarities[top_10]


def search_embeddings(query, top_k, index, model):
    t=time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [fetch_movie_info(idx) for idx in top_k_ids]
    normalized_similarity_scores = (top_k[0].flatten() - min(top_k[0].flatten())) / (max(top_k[0].flatten()) - min(top_k[0].flatten()))
    return results, normalized_similarity_scores


def search_combined(query, top_k, index, model, weight_tfidf, weight_embeddings):
    results_tfidf, scores_tfidf = search_tfidf(query)
    results_embeddings, scores_embeddings = search_embeddings(query, top_k, index, model)
    combined_scores = weight_tfidf * scores_tfidf + weight_embeddings * scores_embeddings
    top_10 = combined_scores.argsort()[:-11:-1]
    return pd.concat([results_tfidf.iloc[top_10], pd.DataFrame(results_embeddings).iloc[top_10]], ignore_index=True)


query="Artificial Intelligence"
print(search_combined(query, top_k=10, index=index, model=model, weight_tfidf=0.5, weight_embeddings=0.5))