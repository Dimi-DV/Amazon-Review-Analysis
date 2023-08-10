
# Amazon Review Analysis

This project focuses on scraping reviews from any Amazon product, converting them into embeddings, clustering these embeddings to group similar reviews, visualizing the clusters, and further analyzing each cluster using transformers and TF-IDF. The goal of the project is to find the most common grievance consumers might have for a product or a range of products. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone this repository.
2. Install the required packages: 
    ```
    pip install -r requirements.txt
    ```
(Note: You might want to create a `requirements.txt` file that lists all the necessary packages.)

## Usage

### Amazon Review Scraper
To scrape Amazon reviews:
```
python amazon_review_bot_v5.py
```

### Embedding and Clustering
To convert reviews into embeddings, cluster them, and visualize the clusters:
```
python embedding_clustering_v7_sentencetransformer.py
```

## Scripts Overview

### amazon_review_bot_v5.py
This script scrapes Amazon product reviews. It can optionally filter reviews based on a specific star rating. The main functionalities include:
- Setting up a Selenium webdriver with randomized user agents to scrape dynamic content.
- Filtering reviews based on a specific star rating.
- Scraping review titles and texts.
- Saving the scraped reviews as a JSON file.

### embedding_clustering_v7_sentencetransformer.py
This script processes the scraped reviews, converts them into embeddings, clusters the embeddings, and performs further analysis. The primary functionalities encompass:
- Loading and preprocessing reviews (tokenization, lemmatization).
- Converting reviews into embeddings using the SentenceTransformer model.
- Clustering the embeddings using KMeans.
- Summarizing each cluster.
- Extracting keywords from the cluster summaries using TF-IDF.
- Visualizing the clusters using t-SNE.

## Results and Visualizations

The results include clustered reviews, summaries for each cluster, and a visual representation of the clusters in the embedding space. The visualization can be interpreted to understand the grouping and similarity of reviews.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 

## Acknowledgements

- Libraries used: BeautifulSoup, Selenium, fake_useragent, gensim, transformers, nltk, sklearn, matplotlib, sentence_transformers, etc.
- Dataset: One star reviews scraped from nine different Amazon speaker products.
