from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the "20 Newsgroups" dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data = newsgroups.data
target = newsgroups.target

# Preprocessing: Tokenization, removing stop words, and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """
    Preprocesses the input text by tokenizing, removing stop words, and lemmatizing.
    :param text: The input text to preprocess.
    :return: The preprocessed text.
    """
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return ' '.join(lemmatized_tokens)

preprocessed_data = [preprocess(text) for text in data]

# Feature Extraction: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(preprocessed_data)
y = target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Model Evaluation
predictions = logistic_model.predict(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, predictions)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
conf_matrix = confusion_matrix(y_test, predictions)

print(accuracy, precision, recall, f1_score)

# Assuming 'conf_matrix' is your confusion matrix and 'newsgroups.target_names' are the labels
labels = newsgroups.target_names

plt.figure(figsize=(15, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
