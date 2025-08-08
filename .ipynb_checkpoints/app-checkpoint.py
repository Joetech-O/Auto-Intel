import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sqlalchemy import create_engine
from collections import Counter
import ast
import altair as alt
from urllib.parse import urlparse

# Database connection
engine = create_engine('postgresql://auto_intel:auto-intel@localhost/auto-intel')

# Cached data loaders
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM newcar_reviews;", engine)
    df['publication_date'] = pd.to_datetime(df['publication_date'], utc=True)
    return df



@st.cache_data
def load_keyword_data():
    keywords=pd.read_sql("SELECT * FROM keyword_pairs;", engine)
    return keywords

# Cached loader with keyword sanitation
@st.cache_data
def load_topic_data():
    df = pd.read_sql("SELECT * FROM news_articles_topics;", engine)

    # Sanitize the topic_keywords column
    def sanitize_keywords(val):
        if not isinstance(val, str):
            return ""
        # Split, strip spaces, lowercase, and remove empty tokens
        keywords = [kw.strip().lower() for kw in val.split(',') if kw.strip()]
        return ', '.join(keywords)

    df['topic_keywords'] = df['topic_keywords'].apply(sanitize_keywords)
    return df


@st.cache_data
def load_ner_data():
    return pd.read_sql("SELECT * FROM car_review_named_entities;", engine)

# Load datasets
df = load_data()
topic_df = load_topic_data()
ner_df = load_ner_data()
keywords=load_keyword_data()

# Sidebar UI
st.sidebar.title("Auto-Intel Dashboard")
option = st.sidebar.radio("Choose Analysis:", (
    "Sentiment Trends", "Source Analysis", "Top Keywords", "Word Cloud", "Topic Modeling", "Named Entities", "Sentiment Timeline"
))
st.title("Auto-Intel News Analysis Dashboard")
# 1. Sentiment Trends
if option == "Sentiment Trends":
    st.subheader("Average Sentiment Score Over Time by Sentiment Type")

    df['month'] = pd.to_datetime(df['publication_date']).dt.to_period('M').dt.to_timestamp()

    trend_df = df.groupby(['month', 'sentiment_label'])['sentiment_score'].mean().reset_index()

    chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X('month:T', title='Month'),
        y=alt.Y('sentiment_score:Q',
                title='Average Sentiment Score',
                scale=alt.Scale(domain=[-1, 1])),  # Force Y-axis to range from -1 to 1
        color=alt.Color('sentiment_label:N', title='Sentiment Type', scale = alt.Scale(
            domain = ["positive", "negative", "neutral"], 
            range=["green", "red", "gray"])),
        tooltip=['month:T', 'sentiment_label:N', 'sentiment_score:Q']
    ).properties(title="Monthly Average Sentiment Score by Type")

    st.altair_chart(chart, use_container_width=True)
# if option == "Sentiment Trends":
#     st.subheader("Average Sentiment Score Over Time by Sentiment Type")

#     df['month'] = pd.to_datetime(df['publication_date']).dt.to_period('M').dt.to_timestamp()

#     # Group by month and sentiment_label
#     trend_df = df.groupby(['month', 'sentiment_label'])['sentiment_score'].mean().reset_index()

#     # Build chart
#     chart = alt.Chart(trend_df).mark_line(point=True).encode(
#         x=alt.X('month:T', title='Month'),
#         y=alt.Y('sentiment_score:Q', title='Average Sentiment Score'),
#         color=alt.Color('sentiment_label:N', title='Sentiment Type'),
#         tooltip=['month:T', 'sentiment_label:N', 'sentiment_score:Q']
#     ).properties(title="Monthly Average Sentiment Score by Type")

#     st.altair_chart(chart, use_container_width=True)

# if option == "Sentiment Trends":
#     st.subheader("Average Sentiment Score Over Time")

#     # Ensure datetime formatting
#     df['month'] = pd.to_datetime(df['publication_date']).dt.to_period('M').dt.to_timestamp()

#     # Group by month: average sentiment score
#     avg_trend_df = df.groupby('month')['sentiment_score'].mean().reset_index()

#     # Altair chart
#     chart = alt.Chart(avg_trend_df).mark_line(point=True).encode(
#         x=alt.X('month:T', title='Month'),
#         y=alt.Y('sentiment_score:Q', title='Avg Sentiment Score'),
#         tooltip=['month:T', 'sentiment_score:Q']
#     ).properties(title="Monthly Average Sentiment Score")

#     st.altair_chart(chart, use_container_width=True)
# if option == "Sentiment Trends":
#     st.subheader("Sentiment Distribution Over Time")

#     df['publication_date'] = pd.to_datetime(df['publication_date'], utc=True)
#     df['month'] = df['publication_date'].dt.to_period('M').dt.to_timestamp()

#     # Group by month + sentiment
#     trend_df = df.groupby(['month', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
#     trend_df = trend_df.set_index('month')

#     st.line_chart(trend_df)

# elif option == "Source Analysis":
#     st.subheader("Sentiment by News Source")
#     sentiment_source = df.groupby(['link', 'sentiment_label']).size().unstack(fill_value=0)
#     st.bar_chart(sentiment_source)

# 2. Source Analysis
elif option == "Source Analysis":
    st.subheader("Sentiment by News Source")

    # Extract domain names from full links
    df['source_domain'] = df['link'].apply(lambda x: urlparse(x).netloc)

    # Group by domain and sentiment
    sentiment_by_source = df.groupby(['source_domain', 'sentiment_label']).size().reset_index(name='count')

    # Optional: Keep only top 10 sources by total article count
    #top_sources = sentiment_by_source.groupby('source_domain')['count'].sum().nlargest(10).index
    #sentiment_by_source = sentiment_by_source[sentiment_by_source['source_domain'].isin(top_sources)]

    # Pivot for plotting
    pivot_df = sentiment_by_source.pivot(index='source_domain', columns='sentiment_label', values='count').fillna(0)

    #re-order 
    pivot_df = pivot_df[[ "positive", "negative", "neutral"]]

    # Plot as stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df.plot(kind='bar', stacked=False, ax=ax, color=[ "green", "red", "gray"])

    plt.xlabel("News Source")
    plt.ylabel("Article Count")
    plt.title("Sentiment Distribution by News Source")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# 3. Top Keywords
elif option == "Top Keywords":
    st.subheader("Top Keywords Frequency")
    if 'phrase' in keywords.columns and 'count' in keywords.columns:
        # Sort by count descending
        top_keywords = keywords.sort_values(by='count', ascending=False)
        # Display table
        st.dataframe(top_keywords)
        # Plot top 30
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_keywords.head(30), x='count', y='phrase', ax=ax)
        ax.set_title("Top 20 Keyword Phrases")
        st.pyplot(fig)
    else:
        st.warning("Required columns 'phrase' or 'count' not found in keyword data.")
# 4. Word Cloud
elif option == "Word Cloud":
    st.subheader("Word Cloud of Keyword Phrases")

    if 'phrase' in keywords.columns and 'count' in keywords.columns:
        # Expand each phrase by its count
        expanded_keywords = []
        for _, row in keywords.iterrows():
            phrase = str(row['phrase'])
            count = int(row['count'])
            expanded_keywords.extend([phrase] * count)

        # Join to form the text input for word cloud
        text = ' '.join(expanded_keywords)

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("The 'phrase' or 'count' column is missing in keyword_pairs.")




# 5. Topic Modeling
elif option == "Topic Modeling":
    st.subheader("Dominant Topics in News Articles")

    # Map topic numbers to descriptions (edit as needed)
    topic_labels = {
        0: "Electric Vehicles",
        1: "Driving Experience",
        2: "Car Design & Features",
        3: "Technology & Innovation",
        4: "Market Trends"
    }

    # Count dominant topics and relabel
    topic_counts = topic_df['dominant_topic'].value_counts().sort_index()
    topic_counts.index = topic_counts.index.map(topic_labels)

    # Display chart
    st.bar_chart(topic_counts)

    # Show topic keywords
    st.subheader("Topic Keywords")
    for i in sorted(topic_df['dominant_topic'].dropna().unique()):
        keywords = topic_df[topic_df['dominant_topic'] == i]['topic_keywords'].iloc[0]
        label = topic_labels.get(i, f"Topic {i}")
        st.markdown(f"**{label}**: {', '.join(word.strip() for word in keywords.split(','))}")



# 6. Named Entity Recognition
elif option == "Named Entities":
    st.subheader("Top Named Entities in News Articles")

    # Label descriptions
    label_descriptions = {
        "ORG": "Organizations (e.g., Tesla, Ford, UN)",
        "GPE": "Geo-Political Entities (e.g., countries, cities like Germany, Nairobi)",
        "PRODUCT": "Commercial Products (e.g., Model 3, iPhone)"
    }

    # Dropdown 1: Choose entity type (ORG, GPE, etc.)
    available_labels = ner_df['label'].dropna().unique().tolist()
    selected_label = st.selectbox("Filter by Entity Type", sorted(available_labels))

    # Description under dropdown
    st.markdown(f"**{selected_label}** → _{label_descriptions.get(selected_label, 'No description available.')}_")

    # Filter by selected label
    filtered_entities = ner_df[ner_df['label'] == selected_label].sort_values(by='count', ascending=False)

    # Bar chart of top entities
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=filtered_entities.head(30), x="count", y="entity", ax=ax)
    ax.set_title(f"Top Entities of Type: {selected_label}")
    st.pyplot(fig)

    # Dropdown 2: Choose specific entity (e.g., Tesla)
    selected_entity = st.selectbox("Select an Entity to Explore Mentions", filtered_entities['entity'].head(30).tolist())

    # Show articles mentioning this entity
    st.markdown(f"### 🔍 Articles Mentioning **{selected_entity}**")

    # Filter articles (assuming 'content' column and clean text)
    mentions_df = df[df['content'].str.contains(selected_entity, case=False, na=False)]

    # Optional: Show sentiment distribution
    sentiment_count = mentions_df['sentiment_label'].value_counts()
    st.bar_chart(sentiment_count)

    # Optional: Display sample articles
    for i, row in mentions_df[['title', 'content']].head(5).iterrows():
        st.markdown(f"**{row['title']}**")
        st.markdown(f"> {row['content'][:300]}...")  # show snippet
        st.markdown("---")
# elif option == "Named Entities":
#     st.subheader("Top Named Entities in News Articles")

#     # Label descriptions
#     label_descriptions = {
#         "ORG": "Organizations (e.g., Tesla, Ford, UN)",
#         "GPE": "Geo-Political Entities (e.g., countries, cities like Germany, Nairobi)",
#         "PRODUCT": "Commercial Products (e.g., Model 3, iPhone)"
#     }

#     # Entity type dropdown
#     available_labels = ner_df['label'].dropna().unique().tolist()
#     selected_label = st.selectbox("Filter by Entity Type", sorted(available_labels))

#     # Show description under dropdown
#     description = label_descriptions.get(selected_label, "No description available.")
#     st.markdown(f"**{selected_label}** → _{description}_")

#     # Filter and display
#     subset = ner_df[ner_df['label'] == selected_label].sort_values(by='count', ascending=False)
#     st.dataframe(subset)

#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=subset.head(30), x="count", y="entity", ax=ax)
#     ax.set_title(f"Top Entities of Type: {selected_label}")
#     st.pyplot(fig)

# Market Trend

elif option == "Sentiment Timeline":
    st.subheader("Monthly Sentiment Score Trend")

    # Load sentiment trend data
    trend_df = pd.read_sql("SELECT * FROM sentiment_trend_monthly", engine)
    trend_df.set_index('publication_date', inplace=True)

    # Line chart for average sentiment
    st.line_chart(trend_df['avg_sentiment'])

    # Stacked sentiment counts
    st.subheader("Monthly Sentiment Distribution")
    bar_data = trend_df[['positive', 'neutral', 'negative']]
    st.bar_chart(bar_data)

    st.markdown("---")

    # Market Trend Visualisation
    st.subheader("Market Trend Analysis")

    market_df = pd.read_sql("SELECT * FROM market_trend_monthly", engine)
    market_df.set_index('publication_date', inplace=True)

    # Line chart for average price and rating
    st.line_chart(market_df[['avg_price', 'avg_rating']])

    # Optional: Bar chart for article volume
    st.subheader("Monthly Article Count")
    st.bar_chart(market_df['article_count'])


# elif option == "Sentiment Timeline":
#     st.subheader("Monthly Sentiment Score Trend")

#     trend_df = pd.read_sql("SELECT * FROM sentiment_trend_monthly", engine)

#     # Line chart for average sentiment
#     st.line_chart(trend_df.set_index('publication_date')['avg_sentiment'])

#     # Stacked sentiment counts
#     st.subheader("Monthly Sentiment Distribution")
#     bar_data = trend_df.set_index('publication_date')[['positive', 'neutral', 'negative']]
#     st.bar_chart(bar_data)
#     plt.xlabel("Months")

