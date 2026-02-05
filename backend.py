import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


def load_and_prep_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None

    score_cols = ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'department' not in df.columns:
        df['department'] = 'General'

    if 'course_name' in df.columns:
        df['lesson_name'] = df['course_name']
    elif 'lesson_name' not in df.columns:
        df['lesson_name'] = 'General'

    df['department'] = df['department'].fillna('General').astype(str)
    df['lesson_name'] = df['lesson_name'].fillna('General').astype(str)

    if 'clean_comment_text' in df.columns:
        df['clean_comment_text'] = df['clean_comment_text'].astype(str)

    if 'has_project' not in df.columns:
        df['has_project'] = df['clean_comment_text'].str.contains('پروژه', na=False).astype(int)
    if 'has_homework' not in df.columns:
        df['has_homework'] = df['clean_comment_text'].str.contains('تمرین|تکلیف|هومورک', na=False).astype(int)
    if 'has_attendance' not in df.columns:
        df['has_attendance'] = df['clean_comment_text'].str.contains('حضور|غیبت|لیست', na=False).astype(int)

    return df


def build_similarity_matrix(df):
    numeric_features = [
        'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6',
        'is_pos', 'is_neg', 'is_neu',
        'has_project', 'has_homework', 'has_attendance'
    ]

    available_numeric = [c for c in numeric_features if c in df.columns]
    agg_dict = {col: 'mean' for col in available_numeric}
    agg_dict['id'] = 'count'  # Count comments

    agg_dict['department'] = lambda x: x.mode()[0] if not x.mode().empty else 'General'
    agg_dict['lesson_name'] = lambda x: ' | '.join(sorted(x.unique().astype(str)))

    prof_profile = df.groupby('professor_name').agg(agg_dict).rename(columns={'id': 'comment_count'})
    prof_profile = prof_profile[prof_profile['comment_count'] >= 5]

    dept_dummies = pd.get_dummies(prof_profile['department'], prefix='dept')

    X = prof_profile[available_numeric].join(dept_dummies).fillna(0)

    sim_matrix = cosine_similarity(X)
    sim_df = pd.DataFrame(sim_matrix, index=prof_profile.index, columns=prof_profile.index)

    return prof_profile, sim_df


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
    if target_prof:
        if target_prof not in sim_df.index: return None
        sim_scores = sim_df[target_prof].sort_values(ascending=False).drop(target_prof).head(50)
        candidates = prof_profile.loc[sim_scores.index].copy()
        candidates['score'] = sim_scores
    else:
        candidates = prof_profile.copy()
        candidates['score'] = candidates['rating_1']

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

    if department and department != "All":
        candidates = candidates[candidates['department'] == department]

    if course_name:
        candidates = candidates[candidates['lesson_name'].str.contains(course_name, case=False, na=False)]

    candidates = candidates.sort_values(by='score', ascending=False)

    cols_to_show = [
        'rating_1', 'rating_3', 'department', 'lesson_name',
        'has_project', 'has_attendance', 'has_homework',
        'comment_count', 'score'
    ]

    available_cols = [c for c in cols_to_show if c in candidates.columns]

    return candidates[available_cols].head(top_n)


def perform_clustering(prof_profile):
    df = prof_profile.copy()

    cluster_features = ['rating_1', 'rating_3', 'has_homework', 'has_project', 'has_attendance']
    available_feats = [c for c in cluster_features if c in df.columns]

    X_cluster = df[available_feats].fillna(0)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    cluster_means = df.groupby('Cluster')['rating_1'].mean().sort_values(ascending=False)
    names = ["The Superstars", "Balanced/Standard", "Strict/Hard", "Lower Ratings"]

    mapping = {}
    for cluster_id, name in zip(cluster_means.index, names):
        mapping[cluster_id] = name

    df['Cluster Name'] = df['Cluster'].map(mapping)
    return df


def get_outliers(prof_profile):
    gems = prof_profile[
        (prof_profile['rating_1'] >= 9.0) &
        (prof_profile['comment_count'] <= 15) &
        (prof_profile['comment_count'] >= 5)
        ].sort_values(by='rating_1', ascending=False).head(5)

    polarizing = prof_profile[
        (prof_profile['has_attendance'] > 0.3) &
        (prof_profile['rating_3'] < 6.0) &
        (prof_profile['rating_1'] > 6.0)
        ].head(5)

    return gems, polarizing
