import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


df_initial_data = pd.read_csv("offer_markets_initial_hours.csv")

print(df_initial_data.head())
all_interests_text = ', '.join(df_initial_data['Interests'].dropna())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_interests_text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Interests')
plt.show()
