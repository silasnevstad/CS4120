import numpy as np
import pandas as pd
from collections import defaultdict
import re
import string
from collections import Counter
import math
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import spacy
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# A few basic stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
             'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
             'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
             'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
             'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
             'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
             'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
             'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
             'now']

# [=== Part A ===]

# Load the dataset
df = pd.read_csv('simpsons_dataset.csv')
df = df.dropna().reset_index(drop=True)

# Initialize a dictionary to hold all spoken words by each character
character_docs = defaultdict(list)


def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase and removing punctuation.

    Args:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text


# Populate the dictionary with character names as keys and concatenated spoken words as values
for index, row in df.iterrows():
    # Check if spoken_words is not NaN
    if pd.notnull(row['spoken_words']):
        processed_text = preprocess_text(row['spoken_words'])
        character_docs[row['raw_character_text']].append(processed_text)

# Combine all spoken words into a single document per character
for character in character_docs:
    character_docs[character] = ' '.join(character_docs[character])


def tokenize(document):
    """
    Tokenizes the input document into words.

    Args:
    document (str): The document to tokenize.

    Returns:
    list: A list of words in the document.
    """
    return document.split()


def calculate_tf(document):
    """
    Calculates term frequency for each word in the document.

    Args:
    document (str): The document to calculate TF for.

    Returns:
    dict: A dictionary of words and their TF values.
    """
    words = tokenize(document)
    word_count = len(words)
    tf = Counter(words)
    for word in tf:
        tf[word] /= word_count
    return tf


def calculate_idf(documents):
    """
    Calculates inverse document frequency for all words across the given documents.

    Args:
    documents (dict): A dictionary of documents with character names as keys.

    Returns:
    dict: A dictionary of words and their IDF values.
    """
    idf = {}
    total_documents = len(documents)
    document_frequency = Counter()

    for document in documents.values():
        unique_words = set(tokenize(document))
        document_frequency.update(unique_words)

    for word, count in document_frequency.items():
        idf[word] = math.log(total_documents / float(count))

    return idf


def calculate_tfidf(documents):
    """
    Calculates TF-IDF for all words in each document.

    Args:
    documents (dict): A dictionary of documents with character names as keys.

    Returns:
    dict: A dictionary with character names as keys and dictionaries of words and their TF-IDF values as values.
    """
    idf = calculate_idf(documents)
    tfidf = defaultdict(dict)

    for character, document in documents.items():
        tf = calculate_tf(document)
        for word, tf_value in tf.items():
            tfidf[character][word] = tf_value * idf.get(word, 0)

    return tfidf


# Extract all unique words from the documents
all_words = set(word for document in character_docs.values() for word in tokenize(document) if len(word) > 3)

# Count how many documents each word appears in
document_count = Counter()
for document in character_docs.values():
    unique_words = set(tokenize(document))
    document_count.update(unique_words.intersection(all_words))

# Filter words that appear in all documents
words_in_all_documents = {word for word, count in document_count.items() if count == len(character_docs)}
# There are no words that appear in all documents? So we will use the top 3 words that appear in most documents

# Calculate the total frequency of these words across all documents
total_word_frequency = Counter()
for document in character_docs.values():
    words = tokenize(document)
    total_word_frequency.update(all_words.intersection(words))

# Find the 3 most common words
most_common_words = total_word_frequency.most_common(3)
most_common_words_extracted = [word for word, count in most_common_words]

# Calculate TF-IDF for all words in each document
tfidf_scores = calculate_tfidf(character_docs)

# Now, let's extract TF-IDF scores for these words from each character document for comparison
tfidf_comparison = defaultdict(dict)

for character in character_docs:
    for word in most_common_words_extracted:
        tfidf_comparison[character][word] = tfidf_scores[character].get(word, 0)

# Print the TF-IDF scores for the top 3 words from the first 5 characters
example_characters = list(tfidf_comparison.keys())[:5]
example_tfidf_scores = {character: tfidf_comparison[character] for character in example_characters}
print(example_tfidf_scores)
# As you can see, the representations are somewhat similar across different characters, but the scores are different.
# This is because the TF-IDF score is a measure of how important a word is to a document in a collection or corpus.
# For example, the word "like" is ever so slightly more important to Lisa Simpson than it is to Miss Hoover.
# The word "well" is quite a lot more important to Edna Krabappel-Flanders than it is to Lisa Simpson or Miss Hoover.


# Preprocess text: tokenization and basic preprocessing for Word2Vec
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


# Function to preprocess text
def preprocess_text_for_word2vec(texts):
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=5000):
        # Tokenize, remove stopwords and non-alphabetic tokens, and lemmatize
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        processed_texts.append(' '.join(tokens))
    return processed_texts


# Applying preprocessing
texts = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])
processed_texts = preprocess_text_for_word2vec(texts)

# Create a new clean dataframe
df_clean = pd.DataFrame({'clean': processed_texts})
df_clean = df_clean.dropna().drop_duplicates()

# Detect and form bigrams
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

# Determine the number of cores to use
cores = multiprocessing.cpu_count()

# Create Word2Vec models
w2v_model_1 = Word2Vec(min_count=20,
                       window=5,
                       vector_size=100,
                       sample=6e-5,
                       alpha=0.03,
                       min_alpha=0.0007,
                       negative=5,
                       workers=cores - 1)

w2v_model_2 = Word2Vec(min_count=20,
                       window=10,
                       vector_size=300,
                       sample=6e-5,
                       alpha=0.03,
                       min_alpha=0.0007,
                       negative=15,
                       workers=cores - 1)

# Build vocabs and train the models
w2v_model_1.build_vocab(sentences, progress_per=10000)
w2v_model_1.train(sentences, total_examples=w2v_model_1.corpus_count, epochs=30, report_delay=1)

w2v_model_2.build_vocab(sentences, progress_per=10000)
w2v_model_2.train(sentences, total_examples=w2v_model_2.corpus_count, epochs=30, report_delay=1)

print(w2v_model_1.wv.most_similar(positive=["homer"]))
print(w2v_model_1.wv.most_similar(positive=["homer_simpson"]))
w2v_model_1.wv.most_similar(positive=["marge"])
w2v_model_1.wv.most_similar(positive=["bart"])
# w2v_model_1.wv.similarity("moe_'s", 'tavern')
w2v_model_1.wv.similarity('maggie', 'baby')
w2v_model_1.wv.similarity('bart', 'nelson')
w2v_model_1.wv.doesnt_match(['jimbo', 'milhouse', 'kearney'])
w2v_model_1.wv.doesnt_match(["nelson", "bart", "milhouse"])
w2v_model_1.wv.doesnt_match(['homer', 'patty', 'selma'])
w2v_model_1.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3)
w2v_model_1.wv.most_similar(positive=["woman", "bart"], negative=["man"], topn=3)

print(w2v_model_2.wv.most_similar(positive=["homer"]))
print(w2v_model_2.wv.most_similar(positive=["homer_simpson"]))
w2v_model_2.wv.most_similar(positive=["marge"])
w2v_model_2.wv.most_similar(positive=["bart"])
# w2v_model_2.wv.similarity("moe_'s", 'tavern')
w2v_model_2.wv.similarity('maggie', 'baby')
w2v_model_2.wv.similarity('bart', 'nelson')
w2v_model_2.wv.doesnt_match(['jimbo', 'milhouse', 'kearney'])
w2v_model_2.wv.doesnt_match(["nelson", "bart", "milhouse"])
w2v_model_2.wv.doesnt_match(['homer', 'patty', 'selma'])
w2v_model_2.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3)
w2v_model_2.wv.most_similar(positive=["woman", "bart"], negative=["man"], topn=3)


def tsnescatterplot(model, word, list_names, size=300):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, size), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    min_comps = min(arrays.shape[0], arrays.shape[1])
    reduc = PCA(n_components=min_comps).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))


tsnescatterplot(w2v_model_1, 'maggie', [i[0] for i in w2v_model_1.wv.most_similar(negative=["maggie"])], size=100)
tsnescatterplot(w2v_model_2, 'maggie', [i[0] for i in w2v_model_2.wv.most_similar(negative=["maggie"])])