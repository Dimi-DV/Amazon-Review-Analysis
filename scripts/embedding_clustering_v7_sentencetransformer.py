import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from summarizer import Summarizer
import os
import glob
from tqdm import tqdm
from collections import Counter
import re
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

class EmbeddingClusterAnalyzer:
    def __init__(self, data_files, word_embedding_file):
        self.data_files = data_files
        self.word_embedding_file = word_embedding_file
        self.reviews = self.load_data(self.data_files)
        self.word2vec = KeyedVectors.load_word2vec_format(word_embedding_file, binary=True)
        self.preprocess_reviews()
        self.document_vectors = self.compute_document_vectors()
        self.document_vectors = self.reduce_dimensionality()
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    def load_data(self, data_files):
        data = []
        for data_file in data_files:
            with open(data_file, 'r') as file:
                file_data = file.read()
            data.extend(json.loads(file_data))
        return data

    def is_english(self, text):
        try:
            text.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def preprocess_reviews(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        english_reviews = [review for review in self.reviews if self.is_english(review["text"])]
        self.reviews = english_reviews
        for review in self.reviews:
            tokens = word_tokenize(review["text"].lower())
            filtered_tokens = [token for token in tokens if token not in stop_words]
            lemmatized_tokens = []
            for token in filtered_tokens:
                pos_tag = nltk.pos_tag([token])[0][1][0].upper()
                pos_tag = wordnet.ADJ if pos_tag == 'J' else \
                    wordnet.VERB if pos_tag == 'V' else \
                        wordnet.ADV if pos_tag == 'R' else \
                            wordnet.NOUN
                lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=pos_tag))
            review["tokenized_text"] = lemmatized_tokens

    def reduce_dimensionality(self, n_components=100):
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(self.document_vectors)
        return reduced_vectors

    def compute_document_vectors(self):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        document_vectors = model.encode([' '.join(review['tokenized_text']) for review in self.reviews])
        return np.array(document_vectors)

    def cluster_reviews(self, n_clusters):
        pairwise_dist = pairwise_distances(self.document_vectors, metric='cosine')
        pairwise_dist = np.where(np.isnan(np.power((1 - pairwise_dist), 2)), 0, pairwise_dist)
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(self.document_vectors)
        self.cluster_labels = clustering.labels_
        return self.cluster_labels

    def count_keywords(self, cluster_summaries, num_keywords=8):
        cluster_top_keywords = []
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        for summary in cluster_summaries:
            # Removing punctuation
            clean_summary = re.sub(r'[^\w\s]', '', summary)

            # Tokenizing and preprocessing the text
            tokens = word_tokenize(clean_summary.lower())
            filtered_tokens = [token for token in tokens if token not in stop_words]
            lemmatized_tokens = []
            for token in filtered_tokens:
                pos_tag = nltk.pos_tag([token])[0][1][0].upper()
                pos_tag = wordnet.ADJ if pos_tag == 'J' else \
                    wordnet.VERB if pos_tag == 'V' else \
                        wordnet.ADV if pos_tag == 'R' else \
                            wordnet.NOUN
                lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=pos_tag))

            # Counting the keywords
            keyword_counts = Counter(lemmatized_tokens)
            top_keywords = keyword_counts.most_common(num_keywords)

            if len(top_keywords) > 0:
                cluster_top_keywords.append(top_keywords)
            else:
                cluster_top_keywords.append([])

        return cluster_top_keywords

    def summarize_cluster(self, reviews, cluster_labels, num_sentences):
        cluster_summaries = []
        for label in set(cluster_labels):
            cluster_reviews = ' '.join(
                [review['text'] for i, review in enumerate(reviews) if cluster_labels[i] == label])
            summaries = []
            chunks = [cluster_reviews[i:i + 1024] for i in range(0, len(cluster_reviews), 1024)]

            for chunk in tqdm(chunks, desc=f"Summarizing cluster {label}", unit="chunk"):
                tokens = self.tokenizer(chunk, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                summary = self.summarizer(chunk, max_length=20, min_length=4, do_sample=False)[0]['summary_text']
                summaries.append(summary)

            with open(f"cluster_{label}_summaries.txt", "w") as f:
                f.write('\n'.join(summaries))

            # Performing keyword counting on the summaries
            summarized_text = ' '.join(summaries)
            top_keywords = self.count_keywords([summarized_text])[0]
            with open(f"cluster_{label}_top_keywords.txt", "w") as f:
                f.write(', '.join([f"{word}: {count}" for word, count in top_keywords if len(top_keywords) > 0]))

            cluster_summaries.append('\n'.join(summaries))

        return cluster_summaries

    def print_cluster_summaries(self, num_sentences=2):
        cluster_summaries = self.summarize_cluster(self.reviews, self.cluster_labels, num_sentences)
        for i, summary in enumerate(cluster_summaries):
            print(f"Cluster {i} summary: {summary}")

    def extract_cluster_keywords(self, num_keywords=5):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        cluster_keywords = []

        for i in range(np.max(self.cluster_labels) + 1):
            cluster_reviews = [review['text'] for j, review in enumerate(self.reviews) if self.cluster_labels[j] == i]
            tfidf_matrix = vectorizer.fit_transform(cluster_reviews)
            feature_names = vectorizer.get_feature_names_out()
            cluster_score = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            keywords = [(feature_names[i], cluster_score[i]) for i in cluster_score.argsort()[::-1]]
            cluster_keywords.append([word for word, score in keywords[:num_keywords]])

        return cluster_keywords


    def visualize_clusters(self):
        tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=2500, random_state=42)
        tsne_data = tsne_model.fit_transform(self.document_vectors)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=self.cluster_labels, cmap="viridis", s=40)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.title("t-SNE visualization of review embeddings with clustering")

        cbar = plt.colorbar(scatter)
        cbar.set_label("Cluster Number")
        centers = [tsne_data[self.cluster_labels == i].mean(axis=0) for i in range(np.max(self.cluster_labels) + 1)]

        for i, center in enumerate(centers):
            plt.annotate(f'Cluster {i}', xy=center, xytext=(center[0], center[1] + 10),
                         arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5),
                         fontsize=12, color='red', weight='bold')

        plt.show()

    def print_cluster_keywords(self, num_keywords=5):
        cluster_keywords = self.extract_cluster_keywords(num_keywords)
        for i, keywords in enumerate(cluster_keywords):
            print(f"Cluster {i} keywords: {', '.join(keywords)}")

# OPTIMAL CLUSTER NUMBER TEST FOR DATASET (OPTIONAL)(NOT ALWAYS RELIABLE)
# ELBOW METHOD: Use number of clusters at the point of the line starting to taper off, the chart will look like an elbow
# SILHOUETTE METHOD: Clusters will range in values between -1 and 1. The closer a cluster is to 1 the more sound it is.
def calculate_wcss(document_vectors):
    wcss = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(document_vectors)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow(wcss):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), wcss)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.show()


def calculate_silhouette_scores(document_vectors):
    silhouette_scores = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        cluster_labels = kmeans.fit_predict(document_vectors)
        silhouette_avg = silhouette_score(document_vectors, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

def plot_silhouette_scores(silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for Different Numbers of Clusters")
    plt.show()

if __name__ == "__main__":
    # Change data_directory to your own
    data_directory = '...'
    # Change the glob parameter from '*.json' to your desired file format
    data_files = glob.glob(os.path.join(data_directory, '*.json'))

    '''You will need to download the embedding file below and place it in the same directory as
    this python script. Alternatively you can download a different vector dataset for the script to use
    as word embedding'''

    word_embedding_file = 'GoogleNews-vectors-negative300.bin'

    embedding_cluster_analyzer = EmbeddingClusterAnalyzer(data_files, word_embedding_file)
    n_clusters = 6
    cluster_labels = embedding_cluster_analyzer.cluster_reviews(n_clusters)

    for i, label in enumerate(cluster_labels):
        print(f"Review {i + 1}: Cluster {label}")

    # OPTIMAL CLUSTER NUMBER TEST FOR DATASET, CALL IT WITH THESE COMMANDS (OPTIONAL)(NOT ALWAYS RELIABLE)
    #wcss = calculate_wcss(embedding_cluster_analyzer.document_vectors)
    #plot_elbow(wcss)

    #silhouette_scores = calculate_silhouette_scores(embedding_cluster_analyzer.document_vectors)
    #plot_silhouette_scores(silhouette_scores)
    # OPTIMAL CLUSTER NUMBER TEST FOR DATASET (OPTIONAL)(NOT ALWAYS RELIABLE)

    embedding_cluster_analyzer.visualize_clusters()
    # Un-comment the line below if you want to use the cluster sentence summaries
    # embedding_cluster_analyzer.print_cluster_summaries()
    embedding_cluster_analyzer.print_cluster_keywords()

