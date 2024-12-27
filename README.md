# Movie Recommendation System

This project implements a **Movie Recommendation System** based on the **IMDb Movies Dataset**. It leverages **Exploratory Data Analysis (EDA)** to gain insights from the data and then applies two types of recommendation techniques:

- **Categorical Recommendation**
- **Content-Based Recommendation**

## Project Overview

The Movie Recommendation System aims to suggest movies to users based on either their category preferences or the content similarity between movies. The project uses the **IMDb Movies Dataset** to train and test the recommendation models.

### Key Features:
1. **Exploratory Data Analysis (EDA)**: Initial analysis of the dataset to uncover trends, distribution, and relationships between variables.
2. **Categorical Recommendation**: Recommends movies based on categorical features such as genre, language, and release year.
3. **Content-Based Recommendation**: Recommends movies based on similarity in movie descriptions, using TF-IDF vectorization to capture textual similarities.

## Libraries Used

To implement the system, the following libraries are used:

```python
import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
```

## Dataset

The dataset used in this project is the **IMDb Movies Dataset**. It contains information about movies, including titles, genres, descriptions, cast, crew, and ratings.

### Dataset Features:
- `movieId`: Unique identifier for each movie.
- `title`: The title of the movie.
- `genres`: Genre(s) of the movie.
- `overview`: Short description or summary of the movie.
- `release_date`: The release date of the movie.
- `rating`: Average user rating of the movie.

## Data Preprocessing & EDA

Before building the recommendation models, extensive data preprocessing and exploratory data analysis (EDA) were performed. Key steps include:

1. **Data Cleaning**: Handling missing values and duplicate entries.
2. **Feature Engineering**: Parsing and transforming text-based data, extracting useful features.
3. **Visualization**: Generating plots to visualize distributions, correlations, and trends (e.g., the distribution of genres, movie ratings over time).

### Some of the visualizations include:
- Wordcloud of frequently used words in movie descriptions.
- Distribution of movie ratings over time.
- Genre-wise distribution of movies.

## Recommendation Models

### 1. **Categorical Recommendation**
This approach recommends movies based on categorical features like genre and release year. Movies with similar genres and release years are grouped together, and recommendations are made based on these similarities.

### 2. **Content-Based Recommendation**
Content-Based Filtering recommends movies based on the similarity of movie descriptions. The textual descriptions of movies are transformed into vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization. The similarity between movies is then calculated using **cosine similarity**.

### Implementation Steps:
1. **Preprocessing**: Tokenize and vectorize movie descriptions using TF-IDF.
2. **Similarity Calculation**: Compute similarity between movies using cosine similarity or linear kernel.
3. **Recommendations**: Given a movie, the system suggests the most similar movies based on content.

## Getting Started

### Installation

To run the project locally, follow the steps below:

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   - Start by loading the dataset.
   - Perform EDA to understand the data.
   - Use the recommendation system functions to get movie suggestions.

### Example Usage

After setting up, you can use the models like this:

```python
# Example: Get movie recommendations based on a movie title
movie_title = "The Dark Knight"
recommended_movies = get_recommended_movies(movie_title)
print(recommended_movies)
```

## Conclusion

This Movie Recommendation System helps users discover movies they might like based on either categorical features or content similarity. The system can be further enhanced by incorporating additional recommendation algorithms and optimizing the models.

