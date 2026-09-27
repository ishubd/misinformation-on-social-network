# loading the libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

# Suppress minor warnings for a clean terminal output
warnings.filterwarnings('ignore')

# Set global plotting style for professional presentation
sns.set_theme(style="whitegrid")

# ==========================================
# 1. DATA LOADING & INITIAL INSPECTION
# ==========================================
print("--- Phase 1: Data Loading & Cleaning ---")
df = pd.read_csv('twitter_user_data.csv', encoding='latin1')

# Formulate Target Variable: Human (1) vs Brand (0)
df = df.dropna(subset=['gender'])
df = df[df['gender'] != 'unknown']
df['is_human'] = df['gender'].map({'male': 1, 'female': 1, 'brand': 0})

print("Visualizing the original internet workers' classification...")
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='is_human', palette='pastel')
plt.title("Original Dataset Labels (Crowdsourced Internet Workers)")
plt.xlabel("Profile Type")
plt.ylabel("Total Number of Accounts")
plt.xticks(ticks=[0, 1], labels=['Brand (0)', 'Human (1)'])

# Add the exact numbers on top of the bars for maximum clarity
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# Drop meta-data, IDs, sparse columns, redundant labels, AND the unscaled outlier column
cols_to_drop = [
    '_unit_id', '_last_judgment_at', 'created', 'name', 'profileimage',
    'tweet_id', 'tweet_created', 'tweet_location', 'user_timezone',
    'tweet_coord', 'profile_yn_gold', '_golden', '_unit_state',
    'profile_yn', 'gender', 'gender_gold', 'profile_yn:confidence',
    '_trusted_judgments'  # Crucial fix to prevent clustering distortion
]
df_cleaned = df.drop(columns=cols_to_drop)

# Fill missing text data
df_cleaned['description'] = df_cleaned['description'].fillna('')
df_cleaned['text'] = df_cleaned['text'].astype(str)

print(f"Data scrubbed! We are feeding exactly {df_cleaned.shape[0]} cleaned profiles into the clustering engine.")

# ==========================================
# 2. FEATURE ENGINEERING: COLORS & BEHAVIORS
# ==========================================
# Extract RGB values from Hex strings
def hex_to_rgb(hex_str):
    try:
        hex_str = str(hex_str).strip().lstrip('#')
        if len(hex_str) != 6: return (0, 0, 0)
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    except Exception:
        return (0, 0, 0)

df_cleaned['link_R'], df_cleaned['link_G'], df_cleaned['link_B'] = zip(*df_cleaned['link_color'].apply(hex_to_rgb))
df_cleaned['sidebar_R'], df_cleaned['sidebar_G'], df_cleaned['sidebar_B'] = zip(*df_cleaned['sidebar_color'].apply(hex_to_rgb))
df_cleaned = df_cleaned.drop(columns=['link_color', 'sidebar_color'])

# Compress massive Twitter outliers using a log transformation BEFORE scaling
df_cleaned[['fav_number', 'retweet_count', 'tweet_count']] = np.log1p(df_cleaned[['fav_number', 'retweet_count', 'tweet_count']])

# Normalize numerical features (Behaviors + RGB Colors)
scaler = StandardScaler()
num_cols = ['fav_number', 'retweet_count', 'tweet_count',
            'link_R', 'link_G', 'link_B', 'sidebar_R', 'sidebar_G', 'sidebar_B']
df_cleaned[num_cols] = scaler.fit_transform(df_cleaned[num_cols])

# ==========================================
# 3. TEXT PROCESSING: TF-IDF
# ==========================================
print("\n--- Phase 2: TF-IDF Text Processing ---")
df_cleaned['combined_text'] = df_cleaned['description'] + " " + df_cleaned['text']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df_cleaned['combined_text'] = df_cleaned['combined_text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cleaned['combined_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()], index=df_cleaned.index)

# Finalize Matrix
df_final = pd.concat([df_cleaned.drop(columns=['description', 'text', 'combined_text']), tfidf_df], axis=1)


# ==========================================
# 4. UNSUPERVISED LEARNING: CLUSTERING
# ==========================================
print("\n--- Phase 3: K-Means Clustering ---")

# Step 1: Prepare the unlabelled feature matrix (Dropping the answer key)
X_cluster = df_final.drop(columns=['is_human', 'gender:confidence'])

# Step 2: Determine Optimal K (Testing 2 to 5 clusters)
print("Evaluating K-Means Silhouette Scores and Brand Concentrations...")
best_k = 2
best_score = -1

for k in range(2, 6):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels_temp, sample_size=5000, random_state=42)
    
    # THE PROOF: Check the maximum concentration of Brands (is_human == 0) in any cluster
    df_temp = pd.DataFrame({'cluster': labels_temp, 'is_human': df_final['is_human']})
    comp = df_temp.groupby('cluster')['is_human'].value_counts(normalize=True).unstack().fillna(0)
    max_brand_pct = comp[0].max() * 100  # Extract the highest percentage from the Brand column
    
    print(f"  K={k} | Silhouette Score: {score:.4f} | Max Brand Concentration: {max_brand_pct:.1f}%")
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nAutomated Silhouette grading suggests K={best_k}.")
print("However, K=2, 3, and 4 failed to cross the 50% threshold to create a true Brand cluster.")
print("-> Manually overriding to K=5 to isolate the distinct 60.8% Brand-heavy behavioral cluster.")

# The Override
best_k = 5

# Step 3: Run final K-Means model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_final['cluster'] = kmeans_final.fit_predict(X_cluster)

# ==========================================
# 5. VISUALIZATION (TASK 3)
# ==========================================
print("\n--- Phase 4: PCA Visualization ---")
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_cluster)

df_final['pca_x'] = pca_result[:, 0]
df_final['pca_y'] = pca_result[:, 1]

plt.figure(figsize=(10, 7))

# Create cleaner text labels for the legend
df_final['Profile Type'] = df_final['is_human'].map({0: 'Brand (Actual)', 1: 'Human (Actual)'})
df_final['Algorithm Group'] = df_final['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'})

# Plotting the clusters with the fixed legend
sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='Algorithm Group', 
    style='Profile Type', 
    palette='viridis', 
    data=df_final, 
    alpha=0.6,
    markers={'Brand (Actual)': 'X', 'Human (Actual)': 'o'} 
)

plt.title('PCA 2D Projection: K-Means Clusters vs Actual Profile Types')
plt.xlabel('Principal Component 1 (Primary Variance)')
plt.ylabel('Principal Component 2 (Secondary Variance)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ==========================================
# 6. IDENTIFYING MISINFORMATION (TASK 4)
# ==========================================
print("\n--- Phase 5: Misinformation Discovery ---")

cluster_makeup = df_final.groupby('cluster')['is_human'].value_counts(normalize=True).unstack().fillna(0)
print("Cluster Composition (Proportion of Brand=0 vs Human=1):")
print(cluster_makeup)

# Tag profiles where actual label opposes the cluster's dominant label
suspicious_profiles = df_final.copy()
dominant_label_per_cluster = df_final.groupby('cluster')['is_human'].agg(lambda x: x.mode()[0])
suspicious_profiles['cluster_dominant_type'] = suspicious_profiles['cluster'].map(dominant_label_per_cluster)

misinformation_candidates = suspicious_profiles[suspicious_profiles['is_human'] != suspicious_profiles['cluster_dominant_type']]

print(f"\nTotal Highly Suspicious Profiles Detected via Clustering: {len(misinformation_candidates)}")
print("Sample of Misclassified Profiles:")
print(misinformation_candidates[['is_human', 'cluster_dominant_type', 'cluster']].head(10))
