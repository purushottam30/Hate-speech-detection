from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time

# Set up Chrome driver
service = Service('path_to_chromedriver')  # Replace with the path to your chromedriver
driver = webdriver.Chrome(service=service)

def get_tweets(username, tweet_count=1000):
    url = f"https://x.com/SadhguruJV"
    driver.get(url)

    tweets = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while len(tweets) < tweet_count:
        # Scroll down to load more tweets
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for page to load
        
        # Parse HTML
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tweet_divs = soup.find_all('div', {'data-testid': 'tweet'})
        
        for tweet_div in tweet_divs:
            tweet_text = tweet_div.get_text(separator=' ')
            if tweet_text not in tweets:
                tweets.append(tweet_text)
                if len(tweets) >= tweet_count:
                    break

        # Check if the page has stopped loading new tweets
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    return tweets[:tweet_count]

if __name__ == "__main__":
    username = 'sadguru'
    tweet_count = 1000

    tweets = get_tweets(username, tweet_count)
    
    with open(f'{username}_tweets.txt', 'w', encoding='utf-8') as f:
        for tweet in tweets:
            f.write(tweet + "\n")
    
    print(f"Saved {len(tweets)} tweets from {username}.")
    driver.quit()
