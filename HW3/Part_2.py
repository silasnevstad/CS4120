from gensim.models import KeyedVectors
from scipy import spatial

# Load pretrained models (converted from GloVe to word2vec format)
# Glove 42B (1.9M vocab, uncased) vs. Glove 840B (2.2M vocab, cased), both are 300 dimensional word vectors
# with varying vocabulary sizes
glove42b = KeyedVectors.load_word2vec_format('glove.42B.300d.word2vec.txt', binary=False)
glove840b = KeyedVectors.load_word2vec_format('glove.840B.300d.word2vec.txt', binary=False)


def convert_glove_to_word2vec(glove_input_file, word2vec_output_file):
    with open(glove_input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Determine the number of dimensions from the first entry
    num_dimensions = len(lines[0].split()) - 1
    vocab_size = len(lines)

    with open(word2vec_output_file, 'w', encoding='utf-8') as f:
        # Write the header
        f.write(f"{vocab_size} {num_dimensions}\n")

        # Write the rest of the file
        for line in lines:
            f.write(line)


def analogy_predict(a, b, c, model):
    # Ensure words are in the model's vocabulary
    if a not in model.index_to_key or b not in model.index_to_key or c not in model.index_to_key:
        return None

    # Calculate the target vector
    target_vector = model[c] + (model[b] - model[a])

    # Find the most similar word, excluding the original words
    most_similar = model.most_similar(positive=[target_vector], negative=[a, b, c], topn=1)

    # Return the most similar word
    return most_similar[0][0]


def load_analogy_questions(file_path):
    categories = [
        "capital-world", "currency", "city-in-state", "family",
        "gram1-adjective-to-adverb", "gram2-opposite",
        "gram3-comparative", "gram6-nationality-adjective"
    ]
    # Create a dictionary with an empty list for each category
    questions = {category: [] for category in categories}
    current_category = None

    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check if the line is a category header
            if line.startswith(": "):
                category = line.strip().split()[1]
                if category in categories:
                    current_category = category
            elif current_category:
                questions[current_category].append(line.strip().split())

    return questions


# Load the analogy questions
analogy_questions = load_analogy_questions('word-test.v1.txt')


def analogy_accuracy(questions, model):
    correct = 0
    total = 0

    for question in questions:
        a, b, c, d = question
        try:
            predicted_d = analogy_predict(a, b, c, model)
            if predicted_d == d:
                correct += 1
            total += 1
        except ValueError:
            # Skip questions with words not in the vocabulary
            continue

    return correct / total if total > 0 else 0


for category, questions in analogy_questions.items():
    accuracy_42b = analogy_accuracy(questions, glove42b)
    accuracy_6b = analogy_accuracy(questions, glove840b)
    print(f"{category}: GloVe 42B accuracy = {accuracy_42b:.2%}, GloVe 840B accuracy = {accuracy_6b:.2%}")

# The GloVe 840B model significantly outperforms the 42B model in most of the analogy tasks. This demonstrates
# the impact of a larger vocabulary (1.9 vs. 2.2 million) and cased vs. uncased, providing more contextual information
# on embedding quality. The larger model far outperforms the smaller model, in capital-world, currency, city-in-state,
# and gram6-nationality-adjective categories, and only slightly outperforms in the gram1-adjective-to-adverb, family and
# gram3-comparative categories. Surprisingly, the 42B model performs better in the gram2-opposite category, which is
# likely due to the uncased nature of the 42B model, allowing it to capture more general semantic relationships.

# Define words and their antonyms
words_and_antonyms = [('increase', 'decrease'), ('enter', 'exit')]


def find_most_similar(words, model):
    for word, antonym in words:
        print(f"Top 10 most similar words to '{word}':")
        similar_to_word = model.most_similar(word, topn=10)
        for similar_word, similarity in similar_to_word:
            print(f"  {similar_word} ({similarity:.4f})")
        print(f"Top 10 most similar words to '{antonym}':")
        similar_to_antonym = model.most_similar(antonym, topn=10)
        for similar_word, similarity in similar_to_antonym:
            print(f"  {similar_word} ({similarity:.4f})")
        print("\n")


print("Using GloVe 42B model:")
find_most_similar(words_and_antonyms, glove42b)

print("Using GloVe 840B model:")
find_most_similar(words_and_antonyms, glove840b)

# The exploration of antonyms reveals that embeddings tend to cluster words with opposite meanings closely,
# highlighting a limitation in capturing semantic opposites solely based on distributional information. For example,
# the word 'increase' is closely associated with 'decrease' in both models, but the antonym 'exit' is not as closely
# associated with 'enter' in either model. This suggests that the models may struggle to capture antonyms with
# different syntactic properties, such as verbs and nouns. The larger GloVe 840B model generally provides more
# accurate and diverse associations for both words and their antonyms, which is consistent with the previous analogy
# task results.
