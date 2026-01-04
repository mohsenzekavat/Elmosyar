import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# ---------------------------------------------------------
# 1. DATA LOADING & PROCESSING
# ---------------------------------------------------------
def load_and_prep_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None

    # Ensure numeric scores
    score_cols = ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- HANDLING DEPT & COURSE ---
    if 'department' not in df.columns:
        df['department'] = 'General'

    # Map 'course_name' to 'lesson_name' (Fix for Search)
    if 'course_name' in df.columns:
        df['lesson_name'] = df['course_name']
    elif 'lesson_name' not in df.columns:
        df['lesson_name'] = 'General'

    # Ensure text columns are strings
    df['department'] = df['department'].fillna('General').astype(str)
    df['lesson_name'] = df['lesson_name'].fillna('General').astype(str)

    if 'clean_comment_text' in df.columns:
        df['clean_comment_text'] = df['clean_comment_text'].astype(str)

    # Keyword Features
    if 'has_project' not in df.columns:
        df['has_project'] = df['clean_comment_text'].str.contains('پروژه', na=False).astype(int)
    if 'has_homework' not in df.columns:
        df['has_homework'] = df['clean_comment_text'].str.contains('تمرین|تکلیف|هومورک', na=False).astype(int)
    if 'has_attendance' not in df.columns:
        df['has_attendance'] = df['clean_comment_text'].str.contains('حضور|غیبت|لیست', na=False).astype(int)

    return df


# ---------------------------------------------------------
# 2. BUILDING THE BRAIN (Now with Department Encoding!)
# ---------------------------------------------------------
def build_similarity_matrix(df):
    # 1. Define Numeric Features (Teaching Style)
    numeric_features = [
        'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6',
        'is_pos', 'is_neg', 'is_neu',
        'has_project', 'has_homework', 'has_attendance'
    ]

    # 2. Aggregation Dictionary
    # Ensure we only aggregate columns that actually exist
    available_numeric = [c for c in numeric_features if c in df.columns]
    agg_dict = {col: 'mean' for col in available_numeric}
    agg_dict['id'] = 'count'  # Count comments

    # Logic: Take most frequent department
    agg_dict['department'] = lambda x: x.mode()[0] if not x.mode().empty else 'General'
    # Logic: Join all unique courses
    agg_dict['lesson_name'] = lambda x: ' | '.join(sorted(x.unique().astype(str)))

    # 3. Create Profile
    prof_profile = df.groupby('professor_name').agg(agg_dict).rename(columns={'id': 'comment_count'})
    prof_profile = prof_profile[prof_profile['comment_count'] >= 5]

    # 4. One-Hot Encoding for Department (The Fix for Scenario 1)
    # This ensures professors in the same department are mathematically "closer"
    dept_dummies = pd.get_dummies(prof_profile['department'], prefix='dept')

    # 5. Combine Style + Topic
    # We join numeric features with department dummies
    X = prof_profile[available_numeric].join(dept_dummies).fillna(0)

    # 6. Build Matrix
    sim_matrix = cosine_similarity(X)
    sim_df = pd.DataFrame(sim_matrix, index=prof_profile.index, columns=prof_profile.index)

    return prof_profile, sim_df


# ---------------------------------------------------------
# 3. HYBRID RECOMMENDER (With All Filters)
# ---------------------------------------------------------
def get_hybrid_recommendation(
        prof_profile,
        sim_df,
        target_prof=None,
        min_score=None,
        project_based=None,
        heavy_homework=None,
        strict_attendance=None,
        department=None,
        course_name=None,
        top_n=5
):
    # 1. Candidate Pool
    if target_prof:
        if target_prof not in sim_df.index: return None
        # Get top 50 similar professors (now smarter thanks to Dept encoding)
        similar_candidates_index = sim_df[target_prof].sort_values(ascending=False).drop(target_prof).head(50).index
        candidates = prof_profile.loc[similar_candidates_index].copy()
    else:
        # Global Search
        candidates = prof_profile.copy()

    # 2. Filters
    if min_score is not None:
        candidates = candidates[candidates['rating_1'] >= min_score]
    if project_based is True:
        candidates = candidates[candidates['has_project'] >= 0.10]
    elif project_based is False:
        candidates = candidates[candidates['has_project'] < 0.10]
    if heavy_homework is True:
        candidates = candidates[candidates['has_homework'] >= 0.20]
    elif heavy_homework is False:
        candidates = candidates[candidates['has_homework'] < 0.20]
    if strict_attendance is True:
        candidates = candidates[candidates['has_attendance'] >= 0.15]
    elif strict_attendance is False:
        candidates = candidates[candidates['has_attendance'] < 0.15]

    # --- Dept & Course Filters ---
    if department and department != "All":
        candidates = candidates[candidates['department'] == department]

    if course_name:
        candidates = candidates[candidates['lesson_name'].str.contains(course_name, case=False, na=False)]

    # 3. Sort & Return
    if target_prof is None:
        candidates = candidates.sort_values(by='rating_1', ascending=False)

    cols_to_show = ['rating_1', 'rating_3', 'department', 'lesson_name', 'has_project', 'has_attendance',
                    'comment_count']
    available_cols = [c for c in cols_to_show if c in candidates.columns]

    return candidates[available_cols].head(top_n)


# ---------------------------------------------------------
# 4. ADVANCED ANALYTICS
# ---------------------------------------------------------
def perform_clustering(prof_profile):
    """
    Groups professors into 4 clusters based on metrics.
    Returns the prof_profile DataFrame with 'Cluster Name' added.
    """
    df = prof_profile.copy()

    # Features for clustering
    cluster_features = ['rating_1', 'rating_3', 'has_homework', 'has_project', 'has_attendance']
    available_feats = [c for c in cluster_features if c in df.columns]

    # Simple imputation
    X_cluster = df[available_feats].fillna(0)

    # Run K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    # Auto-Name Clusters based on Quality Score
    cluster_means = df.groupby('Cluster')['rating_1'].mean().sort_values(ascending=False)
    names = ["The Superstars", "Balanced/Standard", "Strict/Hard", "Lower Ratings"]

    mapping = {}
    for cluster_id, name in zip(cluster_means.index, names):
        mapping[cluster_id] = name

    df['Cluster Name'] = df['Cluster'].map(mapping)
    return df


def get_outliers(prof_profile):
    """
    Returns two dataframes: Underrated Gems and High Risk Professors.
    """
    # 1. Underrated Gems: High Score (>9), Low Reviews (<=15)
    gems = prof_profile[
        (prof_profile['rating_1'] >= 9.0) &
        (prof_profile['comment_count'] <= 15) &
        (prof_profile['comment_count'] >= 5)
        ].sort_values(by='rating_1', ascending=False).head(5)

    # 2. High Risk: Strict Attendance (>0.3) + Low Fairness (<6) + High Quality (>6)
    # (Professors who teach well but are very strict/unfair)
    polarizing = prof_profile[
        (prof_profile['has_attendance'] > 0.3) &
        (prof_profile['rating_3'] < 6.0) &
        (prof_profile['rating_1'] > 6.0)
        ].head(5)

    return gems, polarizing
