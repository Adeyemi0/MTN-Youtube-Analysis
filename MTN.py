# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pytrends.request import TrendReq
from collections import Counter
import nltk
import time
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError, ResponseError

import warnings
# Disable warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("mtn_nigeria_youtube_video_details.csv")

# Define the pages for the Streamlit app
def page_intro():
    
    st.title("MTN Nigeria YouTube Channel Analysis")
    st.markdown("""
    Welcome to the analysis of MTN Nigeria's YouTube channel! This application provides an in-depth exploration of the content and performance metrics for the videos on the channel. By understanding these insights, you can make informed decisions that enhance content strategy, audience engagement, and channel growth.

    ### How to Use This Application
    - **EDA (Exploratory Data Analysis) Page**: This page provides a detailed analysis of video performance metrics such as view count, like count, and comment count. Use these insights to:
        - **Identify High-Performing Content**: Determine which types of videos resonate most with your audience by analyzing metrics like views and likes.
        - **Optimize Content Strategy**: Understand the characteristics of viral videos and replicate successful elements in future content.
        - **Enhance Engagement**: Learn which video tags and keywords drive higher engagement and incorporate them into your video descriptions and titles.
        - **Track Performance Over Time**: Monitor changes in performance metrics over time to evaluate the effectiveness of your content strategy.

    - **Google Trends Page**: This page analyzes trends in search interest related to keywords used in video titles and descriptions. Use these insights to:
        - **Identify Trending Topics**: Discover which topics are gaining interest over time and align your content strategy to capitalize on these trends.
        - **Refine SEO Strategy**: Integrate high-performing keywords into your video metadata to improve discoverability and attract more views.
        - **Understand Audience Interests**: Gain a deeper understanding of what your audience is interested in by analyzing search trends and adjusting your content accordingly.
        - **Plan Future Content**: Use trend data to predict future content needs and plan videos that address upcoming audience interests.

    ### Navigation
    - Use the sidebar to navigate between different pages of the application.
    - Each page provides specific insights and visualizations to help you make data-driven decisions for your YouTube channel.

    We hope this tool helps you maximize the potential of your YouTube content and grow your channel effectively. Happy analyzing!
    """)

    st.subheader("Data Dictionary")
    st.write("""
        Below is the data dictionary for the dataset used in this analysis:

        - **Title:** The title of the YouTube video.
        - **Description:** The description of the YouTube video.
        - **Published At:** The date and time when the video was published.
        - **View Count:** The number of views the video has received.
        - **Like Count:** The number of likes the video has received.
        - **Comment Count:** The number of comments on the video.
        - **Duration:** The duration of the video in seconds.
        - **Caption:** Indicates if the video has captions.
        - **Tags:** Tags associated with the video.
        - **Category ID:** The category ID assigned to the video by YouTube.
        - **Video Link:** The URL of the YouTube video.
    """)

    st.subheader("YouTube Channel Information")
    st.write("""
        - **Channel Title:** MTN Nigeria
        - **Description:** This is the official YouTube Channel for MTN Nigeria where you can find and view our latest videos, products, services and view our latest content. Subscribe to our channel for more interesting, fun, and cool content! ;)
        - **Published At:** 2011-07-08
        - **Subscriber Count:** 79,200
        - **Video Count:** 1,877
        - **View Count:** 89,939,621
    """)

    # Display the channel image
    st.image("https://yt3.googleusercontent.com/ytc/AGIKgqPFA69A_MpA77wstDf8_2d-4I_MV9w7Fc-PclpVGg=s900-c-k-c0x00ffffff-no-rj", 
             caption="MTN Nigeria YouTube Channel", width=300)

def page_eda():
    """
    Page 2: Exploratory Data Analysis
    - Conducts an in-depth analysis of the dataset including visualizations and statistics.
    """
    st.title("Exploratory Data Analysis")

    # Data Overview
    st.subheader("Data Overview")
    st.write("Here's a preview of the dataset:")
    st.write(df.head())

    # Data Types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Caption Analysis
    st.subheader("Caption Analysis")
    st.write("Count of videos with and without captions:")
    st.write(df['Caption'].value_counts())
    # Drop 'Caption' column as it's no longer needed for analysis
    df.drop(columns=['Caption'], inplace=True)

    # Convert 'Published At' column to datetime
    df['Published At'] = pd.to_datetime(df['Published At'], format='%Y-%m-%dT%H:%M:%SZ')
    df['Published At'] = df['Published At'].dt.tz_localize('UTC').dt.tz_convert('Africa/Lagos')
    st.write("Converted 'Published At' to local timezone (Africa/Lagos) for accurate time analysis.")

    # Convert ISO 8601 duration to seconds
    def convert_iso8601_duration(duration):
        """
        Convert ISO 8601 duration to seconds for easier numerical analysis.
        """
        duration = duration.strip('PT')
        hours, minutes, seconds = 0, 0, 0
        if 'H' in duration:
            hours, duration = duration.split('H')
            hours = int(hours)
        if 'M' in duration:
            minutes, duration = duration.split('M')
            minutes = int(minutes)
        if 'S' in duration:
            seconds = int(duration.strip('S'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    df['Duration'] = df['Duration'].apply(convert_iso8601_duration)
    st.write("Converted 'Duration' from ISO 8601 format to seconds for numerical analysis.")

    # Drop 'Category ID' column as it does not add much value to this analysis
    df.drop(columns=['Category ID'], inplace=True)

    # Calculate engagement ratios
    df['Like to View Ratio'] = df['Like Count'] / df['View Count']
    df['Comment to View Ratio'] = df['Comment Count'] / df['View Count']
    st.write("Calculated 'Like to View Ratio' and 'Comment to View Ratio' to measure engagement.")

    # Visualizations
    st.subheader("Engagement Over Time")
    df.set_index('Published At', inplace=True)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(df['View Count'].resample('M').sum(), color='blue', label='View Count')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('View Count', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(df['Like Count'].resample('M').sum(), color='green', label='Like Count')
    ax2.plot(df['Comment Count'].resample('M').sum(), color='red', label='Comment Count')
    ax2.set_ylabel('Like/Comment Count', color='red')
    fig.suptitle('Engagement Over Time')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)
    st.write("Plotted engagement metrics over time to identify trends in viewership and interactions.")

    # Performance by Tags
    st.subheader("Performance by Tags")
    tags_df = df['Tags'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).to_frame('Tag')
    tags_df = tags_df.join(df[['View Count', 'Like Count', 'Comment Count']], how='left')
    text = ' '.join(tags_df['Tag'].dropna())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tag Cloud')
    st.pyplot()
    st.write("Created a word cloud from video tags to visualize common themes.")

    # Performance by Duration
    st.subheader("Performance by Duration")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Duration', y='View Count', hue='Like to View Ratio', size='Comment to View Ratio', sizes=(20, 200))
    st.pyplot()
    st.write("Analyzed performance metrics by video duration to see if length correlates with engagement.")

    # Publishing Time Impact
    st.subheader("Publishing Time Impact")
    df['Hour of Day'] = df.index.hour
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Hour of Day', y='View Count')
    st.pyplot()
    st.write("Investigated the impact of publishing time on view counts.")

    # Keyword Frequency in Titles and Descriptions
    st.subheader("Keyword Frequency in Titles and Descriptions")
    stop_words = set(stopwords.words('english'))
    df['Title'] = df['Title'].astype(str).str.lower().str.replace('mtn', '')
    df['Description'] = df['Description'].astype(str).str.lower().str.replace('mtn', '')
    title_words = ' '.join(df['Title']).split()
    description_words = ' '.join(df['Description']).split()
    title_words = [word for word in title_words if word not in stop_words]
    description_words = [word for word in description_words if word not in stop_words]
    title_word_freq = Counter(title_words)
    description_word_freq = Counter(description_words)
    title_word_freq_df = pd.DataFrame(title_word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False).head(20)
    description_word_freq_df = pd.DataFrame(description_word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False).head(20)
    st.write("Top 20 Keywords in Titles:")
    st.bar_chart(title_word_freq_df.set_index('Word'))
    st.write("Top 20 Keywords in Descriptions:")
    st.bar_chart(description_word_freq_df.set_index('Word'))
    st.write("Identified top keywords in titles and descriptions to understand prevalent themes and topics.")

    # Correlation Matrix
    import warnings
    warnings.filterwarnings('ignore')
    
    # Filter only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(12, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    st.pyplot()
    st.write("Generated a correlation matrix to identify relationships between different metrics.")

     # Reintegrated code starts here
    top_videos_view_count = df.nlargest(10, 'View Count')
    top_videos_like_count = df.nlargest(10, 'Like Count')
    top_videos_comment_count = df.nlargest(10, 'Comment Count')

    # Visualization of Top 10 Videos by View Count
    st.write('### Top 10 Videos by View Count')
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_videos_view_count, x='View Count', y='Title', palette='viridis')
    plt.title('Top 10 Videos by View Count')
    plt.xlabel('View Count')
    plt.ylabel('Video Title')
    st.pyplot(plt.gcf())
    plt.clf()

    # Visualization of Top 10 Videos by Like Count
    st.write('### Top 10 Videos by Like Count')
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_videos_like_count, x='Like Count', y='Title', palette='viridis')
    plt.title('Top 10 Videos by Like Count')
    plt.xlabel('Like Count')
    plt.ylabel('Video Title')
    st.pyplot(plt.gcf())
    plt.clf()

    # Visualization of Top 10 Videos by Comment Count
    st.write('### Top 10 Videos by Comment Count')
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_videos_comment_count, x='Comment Count', y='Title', palette='viridis')
    plt.title('Top 10 Videos by Comment Count')
    plt.xlabel('Comment Count')
    plt.ylabel('Video Title')
    st.pyplot(plt.gcf())
    plt.clf()

    # Define thresholds
    view_threshold = df['View Count'].quantile(0.95)  # Top 5% views
    like_threshold = df['Like Count'].quantile(0.95)  # Top 5% likes
    comment_threshold = df['Comment Count'].quantile(0.95)  # Top 5% comments


    # Categorize videos
    def categorize_video(row):
        if row['View Count'] >= view_threshold:
            return 'Viral Video'
        elif row['View Count'] >= df['View Count'].quantile(0.50):
            return 'Moderately Successful Video'
        else:
            return 'Low Performing Video'

    df['Category'] = df.apply(categorize_video, axis=1)

    # Analyze each category
    category_performance = df.groupby('Category').agg({
        'View Count': 'mean',
        'Like Count': 'mean',
        'Comment Count': 'mean',
        'Duration': 'mean'
    }).reset_index()

    st.write('### Category Performance')
    st.markdown("""
    - **Viral Video**: These are the top-performing videos with the highest view counts, typically in the top 5% of all videos. They capture significant audience attention and can provide insights into what kind of content is most engaging. If you want to understand what makes content viral, pay close attention to these videos.
    - **Moderately Successful Video**: These videos perform better than average but are not quite at the viral level. They fall within the top 50% but below the top 5%. They are good examples of what consistently attracts viewers, helping you understand the type of content that works well over time.
    - **Low Performing Video**: These videos have lower view counts and fall below the median in terms of performance. While they may not have captured as much attention, analyzing these can help identify areas for improvement and avoid content that doesn't resonate as well with the audience.
    """)
    st.dataframe(category_performance)

    # Visualization of category performance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=category_performance, x='Category', y='View Count')
    plt.title('Average View Count by Video Category')
    st.pyplot(plt.gcf())
    plt.clf()

    # Function to analyze tags for a category
    def analyze_tags(data, category):
        category_data = data[data['Category'] == category]
        
        # Explode tags into separate rows
        tags_df = category_data['Tags'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).to_frame('Tag')
        tags_df = tags_df.join(category_data[['View Count']], how='left')
        
        # Calculate average views per tag
        tags_performance = tags_df.groupby('Tag').agg({
            'View Count': ['mean', 'count']
        }).reset_index()
        tags_performance.columns = ['Tag', 'Average Views', 'Count']
        tags_performance = tags_performance.sort_values(by='Average Views', ascending=False)
        
        st.write(f'### Top Tags for {category}')
        st.dataframe(tags_performance.head(10))
        
        # Visualization of top tags by average view count
        plt.figure(figsize=(14, 8))
        top_tags = tags_performance.head(20)
        sns.barplot(data=top_tags, x='Tag', y='Average Views')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top 20 Tags by Average View Count ({category})')
        plt.xlabel('Tags')
        plt.ylabel('Average View Count')
        st.pyplot(plt.gcf())
        plt.clf()
        
        return tags_performance

    # Analyze tags for each category
    st.write('### Tags Analysis')
    viral_tags_performance = analyze_tags(df, 'Viral Video')
    moderate_tags_performance = analyze_tags(df, 'Moderately Successful Video')
    low_tags_performance = analyze_tags(df, 'Low Performing Video')

    # Function to analyze keywords in titles for a category
    def analyze_title_keywords(data, category):
        category_data = data[data['Category'] == category]
        
        # Tokenize titles and remove stopwords
        category_data['Tokens'] = category_data['Title'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])
        
        # Explode tokens into separate rows and remove NaNs
        keywords_df = category_data.explode('Tokens').dropna(subset=['Tokens']).reset_index(drop=True)
        keywords_df = keywords_df.rename(columns={'Tokens': 'Keyword'})
        
        # Calculate average views per keyword
        keywords_performance = keywords_df.groupby('Keyword').agg({
            'View Count': ['mean', 'count']
        }).reset_index()
        keywords_performance.columns = ['Keyword', 'Average Views', 'Count']
        keywords_performance = keywords_performance.sort_values(by='Average Views', ascending=False)
        
        st.write(f'### Top Keywords for {category}')
        st.dataframe(keywords_performance.head(10))
        
        # Visualization of top keywords by average view count
        plt.figure(figsize=(14, 8))
        top_keywords = keywords_performance.head(20)
        sns.barplot(data=top_keywords, x='Keyword', y='Average Views')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top 20 Keywords by Average View Count ({category})')
        plt.xlabel('Keywords')
        plt.ylabel('Average View Count')
        st.pyplot(plt.gcf())
        plt.clf()
        
        return keywords_performance

    # Analyze title keywords for each category
    st.write('### Title Keywords Analysis')
    viral_keywords_performance = analyze_title_keywords(df, 'Viral Video')
    moderate_keywords_performance = analyze_title_keywords(df, 'Moderately Successful Video')
    low_keywords_performance = analyze_title_keywords(df, 'Low Performing Video')

    # Function to extract entities and filter out unwanted ones
    def extract_entities(text):
        doc = nlp(text)
        unwanted_entities = {"9:30pm", "4.30pm", "6pm", "8.30pm", "9pm", "one"}
        return [ent.text for ent in doc.ents if not ent.text.isdigit() 
            and not ent.text.startswith('#') 
            and ent.text.lower() not in unwanted_entities]
    
    # Extract entities from titles and descriptions
    df['Title Entities'] = df['Title'].apply(extract_entities)
    df['Description Entities'] = df['Description'].apply(extract_entities)

    # Flatten the lists of entities into a single series
    title_entities = df['Title Entities'].explode()
    description_entities = df['Description Entities'].explode()

    # Concatenate title and description entities
    all_entities = pd.concat([title_entities, description_entities]).dropna().reset_index(drop=True)

    # Count frequency of each entity
    entity_freq = Counter(all_entities)

    # Convert to DataFrame for plotting
    entity_freq_df = pd.DataFrame(entity_freq.items(), columns=['Entity', 'Frequency']).sort_values(by='Frequency', ascending=False).head(20)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.barplot(data=entity_freq_df, x='Frequency', y='Entity')
    plt.title('Top 20 Named Entities in Titles and Descriptions')
    st.pyplot(plt.gcf())
    plt.clf()

    # Performance by Entities
    def get_entity_performance(entity, df):
        mask = df['Title'].str.contains(entity, case=False, na=False) | df['Description'].str.contains(entity, case=False, na=False)
        entity_df = df[mask]
        return {
            'Entity': entity,
            'Average View Count': entity_df['View Count'].mean(),
            'Average Like Count': entity_df['Like Count'].mean(),
            'Average Comment Count': entity_df['Comment Count'].mean(),
        }

    # Analyze performance of top entities
    top_entities = entity_freq_df['Entity'].tolist()
    entity_performance = [get_entity_performance(entity, df) for entity in top_entities]
    entity_performance_df = pd.DataFrame(entity_performance)

    # Plotting performance of top entities
    plt.figure(figsize=(14, 8))
    sns.barplot(data=entity_performance_df, x='Entity', y='Average View Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average View Count of Top Entities')
    plt.xlabel('Entity')
    plt.ylabel('Average View Count')
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(14, 8))
    sns.barplot(data=entity_performance_df, x='Entity', y='Average Like Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Like Count of Top Entities')
    plt.xlabel('Entity')
    plt.ylabel('Average Like Count')
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(14, 8))
    sns.barplot(data=entity_performance_df, x='Entity', y='Average Comment Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Comment Count of Top Entities')
    plt.xlabel('Entity')
    plt.ylabel('Average Comment Count')
    st.pyplot(plt.gcf())
    plt.clf()


    st.write("### Analysis complete.")

def page_trends():
    st.title("Google Trends Analysis")
    st.markdown("This section analyzes the trends for different keywords over the last five years. Keywords are classified based on their performance. Here's a simple explanation of each category and how you can use the information to make informed decisions.")

    # Explanation of keyword classifications
    st.subheader("Keyword Classification Explanation")
    st.markdown("""
    - **Trending Keywords**: Keywords that have consistently grown in popularity. Focus on these for future-proof strategies.
    - **Stable and Increasing**: Keywords with steady popularity that is slowly increasing. Reliable for long-term content planning.
    - **Stable but Decreasing**: Keywords that have maintained stable popularity but show a slight decline. Good for short-term content.
    - **Relatively Stable**: Keywords with neither significant increases nor decreases in popularity. Suitable for evergreen content.
    - **Declining**: Keywords gradually losing popularity. Use them cautiously if still relevant to your audience.
    - **Significantly Decreasing**: Keywords with a sharp drop in interest. Best to avoid as they are becoming obsolete quickly.
    - **Cyclically Interested**: Keywords that peak in popularity at regular intervals, like seasonal trends. Plan content around these peaks for maximum impact.
    """)

    # Explanation of practical use
    st.markdown("""
    **How to Use This Information**:
    - **Trending and Increasing**: Prioritize these keywords for content that will remain relevant in the future.
    - **Stable**: Great for creating content that is consistently relevant over a long period.
    - **Decreasing**: Use cautiously for content that may still resonate with your current audience but avoid long-term reliance.
    - **Cyclically Interested**: Plan your content schedule to align with peaks in interest to maximize engagement.
    """)

    # Request keywords for trend analysis
    keywords = st.text_area("Enter keywords to compare, separated by commas:", "MTN Nigeria, Airtel Nigeria")
    keywords = [keyword.strip() for keyword in keywords.split(',')]

    if st.button("Analyze Trends"):
        pytrends = TrendReq(hl='en-US', tz=360)
        retry_attempts = 5  # Set the number of retry attempts
        delay = 10  # Delay between attempts in seconds

        for attempt in range(retry_attempts):
            try:
                pytrends.build_payload(keywords, cat=0, timeframe='today 5-y', geo='NG', gprop='')

                # Interest over time
                interest_over_time = pytrends.interest_over_time()
                if not interest_over_time.empty:
                    st.line_chart(interest_over_time.drop(columns=['isPartial']))

                    # Interest by region
                    st.subheader("Interest by Region")
                    interest_by_region = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
                    st.write(interest_by_region)

                    # Related queries
                    st.subheader("Related Queries")
                    related_queries = pytrends.related_queries()
                    for kw in keywords:
                        st.write(f"Top queries for '{kw}':")
                        st.write(related_queries[kw]['top'])
                        st.write(f"Rising queries for '{kw}':")
                        st.write(related_queries[kw]['rising'])
                else:
                    st.write("No trends data found for the selected keywords. Try different keywords or check back later.")
                break  # Exit the loop if successful

            except TooManyRequestsError:
                if attempt < retry_attempts - 1:
                    st.error(f"Too many requests. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retry_attempts})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    st.error("Too many requests. Please try again later.")
            except ResponseError as e:
                st.error(f"Response error: {e}. Please check your keywords and try again.")
                break  # Exit the loop if another error occurred
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}. Please try again later.")
                break  # Exit the loop if another error occurred

# Streamlit app main navigation
page = st.sidebar.selectbox("Choose a page", ["Introduction", "EDA", "Google Trends"])

if page == "Introduction":
    page_intro()
elif page == "EDA":
    page_eda()
elif page == "Google Trends":
    page_trends()
