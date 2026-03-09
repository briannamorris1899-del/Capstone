import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure visuals folder exists
os.makedirs('visuals', exist_ok=True)

# Load cleaned data
df = pd.read_csv('data/cleaned/epilepsy_cleaned.csv')

# 1. Seizure Trends Over Time
plt.figure(figsize=(12,5))
df.groupby('SeizureDate').size().plot()
plt.title('Seizure Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Seizures')
plt.tight_layout()
plt.savefig('visuals/seizure_trends.png')
plt.close()

# 2. Correlation Heatmap (numeric columns only)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('visuals/correlation_heatmap.png')
plt.close()

# 3. Sleep vs Seizures Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='SleepDeviation', y='SeizureCount')
plt.title('Sleep Deviation vs Seizure Count')
plt.tight_layout()
plt.savefig('visuals/sleep_vs_seizures_scatter.png')
plt.close()

# 4. Demographic Breakdown
plt.figure(figsize=(10,6))
df.groupby('AgeGroup').size().plot(kind='bar')
plt.title('Seizures by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Seizures')
plt.tight_layout()
plt.savefig('visuals/demographic_breakdown.png')
plt.close()

