import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gc

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# ================= LOAD DATA ==================================
# load wiki_movie_plots_deduped.csv top 500 rows into a DataFrame
data = pd.read_csv('wiki_movie_plots_deduped.csv', nrows=500)
df = data[['Title','Plot']]
del data
gc.collect()
df.dropna(inplace=True)
df.drop_duplicates(subset=['Plot'],inplace=True)
print('done loading data')
# ===============================================================




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
# ===============================================================



# ================== TF-IDF =====================================
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['processed_plot'])

# print(tfidf_matrix)
print("done tf-idf")
# =======================================================


# ===================== search function ==================
from sklearn.metrics.pairwise import linear_kernel

def search(query):
    # Preprocess the query
    query = preprocess_text(query)

    # Convert the query to its corresponding TF-IDF representation
    query_tfidf = vectorizer.transform([query])

    # Compute the cosine similarity between the user query and all the movie plots
    cosine_similarities = linear_kernel(query_tfidf, tfidf_matrix).flatten()
    print(cosine_similarities)

    # Get the top 10 most similar movie plots
    top_10 = cosine_similarities.argsort()[:-11:-1]

    # Return the titles of the top 10 most similar movies
    return df['Title', 'processed_plot'].iloc[top_10]

print(search('saloon bartender'))
