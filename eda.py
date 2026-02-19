import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
cols = ['id', 'article_id', 'keyword', 'country', 'text', 'label']
df = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t', names=cols, skiprows=4)

#Data Cleaning
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

#Define Binary Task 
df['is_pcl'] = df['label'].apply(lambda x: 1 if x >= 2 else 0)

#PCL Density per Keyword ---
plt.figure(figsize=(12, 6))
keyword_counts = df.groupby('keyword')['is_pcl'].value_counts(normalize=True).unstack()
keyword_counts[1].sort_values().plot(kind='barh', color='salmon')
plt.title('Percentage of PCL Instances by Keyword Context')
plt.xlabel('Proportion of Positive PCL Cases')
plt.tight_layout()
plt.show()

#Label Distribution (0-4) ---
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribution of Original Annotator Labels (0-4)')
plt.xlabel('Label Value')
plt.ylabel('Count')
plt.show()

#Additional: Text Length Analysis ---
df['word_count'] = df['text'].str.split().str.len()
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='word_count', hue='is_pcl', common_norm=False, fill=True)
plt.title('Distribution of Word Counts: PCL vs No-PCL')
plt.xlim(0, 400) 
plt.show()

#Summary Statistics Table ---
summary = df.groupby('keyword').agg({
    'id': 'count',
    'is_pcl': 'sum'
}).rename(columns={'id': 'Total', 'is_pcl': 'PCL_Positive'})
summary['PCL_%'] = (summary['PCL_Positive'] / summary['Total']) * 100
print(summary.sort_values('PCL_%', ascending=False))