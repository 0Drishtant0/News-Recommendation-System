from django.shortcuts import render
import pandas as pd
import pyarrow as pa
import joblib

movies_data = pd.read_parquet("static/top_2k_movie_data.parquet")
titles = movies_data['title']
article_titles_small = titles.to_list()

#using this small data for autocomplete suggestions
articles = pd.read_parquet("static/small_data.parquet")
article_titles_small = articles['title'].tolist()

# Load the model and the data
Amodel = joblib.load('static/article_model.joblib')
articles_df = joblib.load('static/articles_df.pkl')
vectorizer = joblib.load('static/vectorizer.pkl')


##processing the input 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def process_text(text):
    """Process text function.
    Input:
        text: a string containing the text
    Output:
        processed_text: a list of words containing the processed text

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks    
    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    # tokenize text
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text_tokens = tokenizer.tokenize(text)

    processed_text = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # processed_text.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            processed_text.append(stem_word)

    return processed_text




## get recommendations
def find_related_articles(keywords):
    # Preprocess the user input keywords
    processed_keywords = process_text(keywords)
    processed_keywords = ' '.join(processed_keywords)
    
    # Transform the user input keywords into TF-IDF features
    keyword_features = vectorizer.transform([processed_keywords])
    
    # Find the nearest neighbors to the user input keywords
    distances, indices = Amodel.kneighbors(keyword_features)
    
    # Get the related articles based on the nearest neighbors
    related_articles = articles_df.iloc[indices[0]]
    
    return related_articles







def news_detail(request, news_id):

    article = articles_df.loc[articles_df['article_id'] == news_id]
    article = article.to_dict()
    return render(request, 'recommender/news_detail.html', {
        'article': article
        })
                    



def main(request):

    global  model, article_titles_small
 
    if request.method == 'GET':
       
        return render(
                request,
                'recommender/index.html',
                {
                    'all_movie_names':article_titles_small,
                    'input_provided':'',
                    'movie_found':'',
                    'recomendation_found':'',
                    'recommended_movies':[],
                    'input_movie_name':''
                }
            )

    if request.method == 'POST':

        data = request.POST
        movie_name = data.get('movie_name') ## get movie name from the frontend input field

    
        final_recommendations = find_related_articles(movie_name)
       
        if not final_recommendations.empty:
            recommended_movies_list = [row.to_dict() for _, row in final_recommendations.iterrows()]
            return render(
                request,
                'recommender/result.html',
                {
                    'all_movie_names':article_titles_small,
                    'input_provided':'yes',
                    'movie_found':'yes',
                    'recomendation_found':'yes',
                    'recommended_movies':recommended_movies_list,
                    'input_movie_name':movie_name
                }
            )
        else:
            return render(
                request,
                'recommender/index.html',
                {
                    'all_movie_names':article_titles_small,
                    'input_provided':'yes',
                    'movie_found':'',
                    'recomendation_found':'',
                    'recommended_movies':[],
                    'input_movie_name':movie_name
                }
            )

