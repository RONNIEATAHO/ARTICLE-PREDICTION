from newspaper import Article

# Example URL (you can change this to any online news article)
url = 'https://www.bbc.com/news/world-66236083'

# Download and parse the article
article = Article(url)
article.download()
article.parse()

# Print article title and text
print("Title:", article.title)
print("\nFull Text:\n", article.text)
