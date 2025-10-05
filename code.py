# ====================================================================
# PROJECT CONFIGURATION (REPLACE THESE PLACEHOLDERS)
# ====================================================================

# Replace with your actual GitHub credentials
GITHUB_USERNAME = "Your_GitHub_Username"           # <-- REPLACE THIS
GITHUB_REPO_NAME = "Netflix-Clustering-Capstone"   # <-- REPLACE THIS (Must match your repo name)
GITHUB_PAT = "Your_Personal_Access_Token"          # <-- REPLACE THIS with your PAT

# File names for the project
LOCAL_FOLDER = "netflix_capstone_repo"
NOTEBOOK_FILE = "Sample_EDA_Submission_Template.ipynb"
DATA_FILE = "NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv"


# ====================================================================
# GITHUB SETUP AND FILE ORGANIZATION
# ====================================================================

import os
import sys

# Exit if the data file isn't present
if not os.path.exists(DATA_FILE):
    print(f"ERROR: Dataset file '{DATA_FILE}' not found. Please upload it.")
    sys.exit(1)

print("--- STARTING GITHUB AND ENVIRONMENT SETUP ---")
# Install Git and set user identity
!apt-get update > /dev/null
!apt-get install git -y > /dev/null
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"

# Create folder and organize files
os.makedirs(LOCAL_FOLDER, exist_ok=True)
try:
    !cp "{NOTEBOOK_FILE}" "{LOCAL_FOLDER}/"
    !cp "{DATA_FILE}" "{LOCAL_FOLDER}/"
    print("Files organized successfully.")
except Exception as e:
    print(f"WARNING: Could not copy files. Ensure files exist. Error: {e}")

# Change directory into the project folder
os.chdir(LOCAL_FOLDER)
print(f"Current directory changed to: {os.getcwd()}")


# ====================================================================
# NETFLIX CAPSTONE PROJECT CODE (EDA, WRANGLING, & MODEL)
# ====================================================================

# Import All Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

sns.set_style('whitegrid')

# Load the dataset
try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    print(f"FATAL ERROR: Could not load the dataset inside the working directory. {e}")
    sys.exit(1)


print("\n--- 1. DATA WRANGLING AND FEATURE ENGINEERING ---")

# --- Wrangling ---
for col in ['director', 'cast', 'country']:
    df[col].fillna('Unknown', inplace=True)
df['date_added'].fillna(df['date_added'].mode()[0], inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)

# --- Feature Engineering ---
df['year_added'] = pd.to_datetime(df['date_added'].str.strip()).dt.year.astype('Int64')
df['main_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())
df.rename(columns={'listed_in': 'genre'}, inplace=True)
df['is_movie'] = df['type'].apply(lambda x: 1 if x == 'Movie' else 0)
df['duration_int'] = df['duration'].str.extract('(\d+)').astype(int)
df['duration_type'] = df['duration'].apply(lambda x: 'min' if 'min' in x else 'Season')
df['primary_genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip())
df['is_known_country'] = df['main_country'].apply(lambda x: 'Known' if x != 'Unknown' else 'Unknown')


# ====================================================================
# 3. CORE EDA VISUALIZATIONS (Examples)
# ====================================================================
print("\n--- 2. GENERATING KEY EDA VISUALIZATIONS ---")

# --- Chart 1: Content Type Distribution ---
content_counts = df['type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(content_counts, labels=content_counts.index, autopct='%1.1f%%', startangle=90, colors=['#E50914', '#221e1f'], wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Content Type (Movies vs. TV Shows)', fontsize=16)
plt.ylabel('')
plt.show()

# --- Chart 11: Content Type Acquisition Trend by Top 3 Countries (FIXED) ---
top_3_countries = df[df['main_country'] != 'Unknown']['main_country'].value_counts().head(3).index.tolist()
trend_df = df[df['main_country'].isin(top_3_countries)]
plot_data_11 = trend_df.groupby(['year_added', 'main_country', 'type']).size().reset_index(name='Title_Count')
plt.figure(figsize=(14, 7))
sns.lineplot(x='year_added', y='Title_Count', hue='main_country', style='type', data=plot_data_11, palette='husl', linewidth=2, marker='o')
plt.title('Acquisition Trend of Titles by Type and Top 3 Countries', fontsize=16)
plt.xlabel('Year Added', fontsize=12)
plt.ylabel('Number of Titles Added', fontsize=12)
plt.legend(title='Country & Type')
plt.show()

# --- Chart 13: Acquisition Trend of Unknown vs. Known Country Titles (FIXED) ---
plot_data_13 = df.groupby(['year_added', 'is_known_country']).size().reset_index(name='Title_Count')
plt.figure(figsize=(12, 6))
sns.lineplot(x='year_added', y='Title_Count', hue='is_known_country', data=plot_data_13, palette={'Known': '#221e1f', 'Unknown': '#E50914'}, linewidth=2, marker='o')
plt.title('Acquisition Trend: Known vs. Unknown Country of Origin', fontsize=16)
plt.xlabel('Year Added', fontsize=12)
plt.ylabel('Number of Titles Added', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Country Status')
plt.show()

# --- Chart 14: Correlation Heatmap ---
numerical_df = df[['release_year', 'year_added', 'duration_int', 'is_movie']]
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.show()


# --- Chart 15: Pair Plot (Sampled) ---
numerical_for_pairplot = df[['release_year', 'year_added', 'duration_int', 'is_movie']]
sns.pairplot(numerical_for_pairplot.sample(n=min(len(numerical_for_pairplot), 2000), random_state=42), 
             hue='is_movie', palette={1: '#E50914', 0: '#221e1f'}, plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Numerical Features Colored by Content Type (Sampled)', y=1.02, fontsize=16)
plt.show()


# --- Machine Learning Model Setup ---
df_model = df.drop(columns=['show_id', 'title', 'director', 'cast', 'description', 'date_added', 'country', 'duration', 'genre', 'duration_type'])
X = df_model.drop('is_movie', axis=1)
y = df_model['is_movie']
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n--- 3. TRAINING CLASSIFICATION MODEL ---")
model.fit(X_train, y_train)


# ====================================================================
# 5. FINAL GITHUB PUSH (The Submission Step)
# ====================================================================

print("\n--- PUSHING TO GITHUB ---")

# Link remote origin using the tokenized URL for authentication
REMOTE_URL = f"https://{GITHUB_USERNAME}:{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}.git"

# Initialize git, stage, commit, and push
!git init > /dev/null
!git remote remove origin 2> /dev/null
!git remote add origin "{REMOTE_URL}"

# Commit (Note: This saves the final state of the notebook file)
COMMIT_MESSAGE = "Final EDA Capstone Submission with Analysis and Code Fixes"
!git add .
!git commit -m "{COMMIT_MESSAGE}"

# Push the local commits to the main branch
!git push -u origin main

print("\n✅ PROCESS COMPLETE. Please check your GitHub repository!")
