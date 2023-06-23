import pickle

# load tfidf_matrix.pickle as variable
with open('tfidf_matrix.pickle', 'rb') as handle:
    tfidf_matrix = pickle.load(handle)

print(tfidf_matrix)