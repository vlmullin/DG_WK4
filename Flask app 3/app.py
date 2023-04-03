from flask import Flask, render_template, request
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

app = Flask(__name__)

# Load data into pandas dataframes
movies_df = pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/movies.csv')
ratings_df = pd.read_csv('https://raw.githubusercontent.com/vlmullin/DG_WK4/main/small/ratings.csv')

# Define the genre options
genre_options = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Define the SVD model
reader = Reader()
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Define the home page
@app.route('/')
def home():
    return render_template('home.html', genre_options=genre_options)

# Define the recommendation page
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the user ID and genre from the form
    user_id = request.form['userId']
    genre = request.form['genre']

    # Check if the user ID is valid
    if not user_id.isdigit() or int(user_id) < 1 or int(user_id) > 610:
        return render_template('error.html', message='Invalid user ID. Please enter a number between 1 and 610.')

    # Check if the genre is valid
    if genre not in genre_options:
        return render_template('error.html', message='Invalid genre. Please select a genre from the dropdown menu.')

    # Filter the data by user ID and genre
    user_ratings = ratings_df.loc[ratings_df['userId'] == int(user_id)]
    genre_movies = movies_df.loc[movies_df['genres'].str.split('|').apply(lambda x: genre in x)]

    # Filter the genre movies by unrated movies by the user
    unrated_movies = genre_movies[~genre_movies['movieId'].isin(user_ratings['movieId'])]

    # Use the SVD model to predict the ratings for the unrated movies
    predictions = []
    for movie_id in unrated_movies['movieId'].unique():
        predictions.append((movie_id, svd.predict(int(user_id), movie_id).est))
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get the top 10 predicted movies
    recommended_movies = []
    for i, (movie_id, _) in enumerate(predictions):
        recommended_movies.append(genre_movies.loc[genre_movies['movieId'] == movie_id]['title'].values[0])
        if i == 9:
            break

    return render_template('recommendations.html', user_id=user_id, genre=genre, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
