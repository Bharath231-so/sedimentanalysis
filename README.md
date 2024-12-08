# sedimentanalysis
1.Certainly! Here's a Python code that outlines the steps involved in stock movement analysis based on social media sentiment. We'll break it down into parts: data collection (using Twitter as an example), sentiment analysis, and stock price prediction.

We'll use the following libraries:

Tweepy for Twitter API integration. TextBlob for sentiment analysis. yfinance for retrieving historical stock prices. pandas and numpy for data manipulation. sklearn for machine learning models and evaluation. Prerequisites: Install required libraries:

pip install tweepy textblob  yfinance pandas numpy scikit-learn

2.Data Collection (Twitter API + Stock Data) First, you'll need Twitter API keys. You can get them by creating a Twitter developer account and setting up an application. Then you can access the keys (API Key, API Secret Key, Access Token, and Access Token Secret).

# Twitter API credentials (replace with your own)
API_KEY = 'p8J9oJWQUzhbfpTGdXSoFc4oD'
API_SECRET_KEY = 'n8mA7NCRLapaN4LBc7eNQaHVDuCqSvAef6qAUa2lndT8unEK17'
ACCESS_TOKEN = '1864913576529375232-jREMGxf5sgxp4hH1KccHEkadySZHl1'
ACCESS_TOKEN_SECRET = 'TyUMQkB74EO8rkmukwnwch50alSEiPMXxmVjGwxHGcKpU'

# Set up the Twitter API client
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def fetch_tweets(query, count=100):
    # Fetch tweets based on the query (stock symbol or company name)
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count)
    tweet_data = []

    for tweet in tweets:
        tweet_data.append({'text': tweet.full_text, 'created_at': tweet.created_at})

    return pd.DataFrame(tweet_data)

# Fetch stock price data from Yahoo Finance
def fetch_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Example usage
stock_symbol = 'AAPL'
tweets = fetch_tweets(stock_symbol, count=200)  # Fetch 200 recent tweets
stock_data = fetch_stock_data(stock_symbol, '2023-01-01', '2023-11-01')  # Historical stock data

# Display the data
print(tweets.head())
print(stock_data.head())

#Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Return polarity score: positive (>0), negative (<0), or neutral (==0)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis on the tweets DataFrame
tweets['sentiment'] = tweets['text'].apply(analyze_sentiment)

# Count the number of positive, negative, and neutral sentiments
sentiment_counts = tweets['sentiment'].value_counts()
print(sentiment_counts)

# Create sentiment score (positive=1, negative=-1, neutral=0)
def sentiment_score(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    else:
        return 0

tweets['sentiment_score'] = tweets['sentiment'].apply(sentiment_score)

# Group by date and calculate the daily sentiment score
tweets['date'] = tweets['created_at'].dt.date
daily_sentiment = tweets.groupby('date')['sentiment_score'].sum().reset_index()

# Merge the sentiment data with the stock price data
stock_data['Date'] = stock_data.index.date

Sentiment Analysis Next, we will analyze the sentiment of each tweet using TextBlob. We'll classify the sentiment as positive, negative, or neutral.

from IPython import get_ipython
from IPython.display import display
from textblob import TextBlob  # Make sure TextBlob is imported here as well
import pandas as pd

# Get the output of the previous cell (ipython-input-1-2cb3f33fe95c)
_prev_cell_output = get_ipython().user_ns.get('_', None)

# If the previous output is a DataFrame and likely 'tweets', use it
if isinstance(_prev_cell_output, pd.DataFrame):
    tweets = _prev_cell_output

# If 'tweets' is still not found, it may need to be reloaded/recreated
# Example: tweets = pd.read_csv('tweets.csv')  # If it was saved to a file
# or tweets = fetch_tweets(stock_symbol, count=200)  # Refetch tweets

# Redefine the analyze_sentiment function in this cell
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Return polarity score: positive (>0), negative (<0), or neutral (==0)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Now apply the sentiment analysis
tweets['sentiment'] = tweets['text'].apply(analyze_sentiment)

3.Feature Engineering: Sentiment Score & Lagged Features We’ll aggregate sentiment scores for each day and create lagged features to help with stock price prediction.

# Create sentiment score (positive=1, negative=-1, neutral=0)
def sentiment_score(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    else:
        return 0

tweets['sentiment_score'] = tweets['sentiment'].apply(sentiment_score)

# Group by date and calculate the daily sentiment score
tweets['date'] = tweets['created_at'].dt.date
daily_sentiment = tweets.groupby('date')['sentiment_score'].sum().reset_index()

# Merge the sentiment data with the stock price data
stock_data['Date'] = stock_data.index.date
stock_data = pd.merge(stock_data, daily_sentiment, how='left', left_on='Date', right_on='date')

# Fill missing sentiment scores with 0 (no sentiment for that day)
stock_data['sentiment_score'].fillna(0, inplace=True)
# Create lagged sentiment features
stock_data['sentiment_score_lag1'] = stock_data['sentiment_score'].shift(1)
stock_data['sentiment_score_lag2'] = stock_data['sentiment_score'].shift(2)

# Drop the date column as it's not needed anymore
stock_data.drop(['date'], axis=1, inplace=True)

# Display the data
print(stock_data.tail())

4.Stock Price Prediction Model (Classification) Now, we will use machine learning (Random Forest in this case) to predict the stock price movement (up or down) based on sentiment and other features.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Calculate the daily stock price change
stock_data['price_change'] = stock_data['Close'].pct_change()

# Create the target variable (1 for price increase, 0 for price decrease)
stock_data['target'] = np.where(stock_data['price_change'] > 0, 1, 0)

# Drop missing values
stock_data.dropna(subset=['sentiment_score_lag1', 'sentiment_score_lag2', 'target'], inplace=True)

# Features and target variable
X = stock_data[['sentiment_score_lag1', 'sentiment_score_lag2']]
y = stock_data['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


5.Evaluation Finally, we evaluate the model’s performance using accuracy and classification metrics (precision, recall, F1-score). If the model is successful, you can use it to make real-time predictions based on new social media data.

6.Additional Considerations: Real-time prediction: For real-time prediction, you could set up a system to fetch new tweets and stock data periodically (e.g., every day) and run predictions. Advanced models: You can use more advanced models (e.g., LSTM or XGBoost) for better performance, especially for time-series forecasting. Conclusion: This code is a basic implementation of stock movement analysis based on social media sentiment. You can extend it by incorporating more features (such as volume of mentions), using different machine learning models, and refining sentiment analysis. Also, you can backtest this system to validate its predictions and profitability.



    
