import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend import load_and_prep_data, build_similarity_matrix, get_hybrid_recommendation, perform_clustering, get_outliers

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Professor Intelligence", page_icon="🎓", layout="wide")


@st.cache_data
def get_data():
    # Make sure this points to your file!
    df = load_and_prep_data('data/processed/sentiment_data.csv')
    if df is None: return None, None, None
    prof_profile, sim_df = build_similarity_matrix(df)
    return df, prof_profile, sim_df


df, prof_profile, sim_df = get_data()

prof_profile = perform_clustering(prof_profile)

if df is None:
    st.error("Error: 'sentiment_data.csv' not found.")
    st.stop()

# ---------------------------------------------------------
# 2. NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Professor AI")
page = st.sidebar.radio("Navigate", ["Overview", "Search & Filter", "Professor Profile", "Compare", "Recommender"])
st.sidebar.divider()
st.sidebar.info(f"Database: {len(df)} Comments\nProfessors: {len(prof_profile)}")

# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.title("University Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Opinions", f"{len(df):,}")
    c2.metric("Active Professors", f"{len(prof_profile)}")
    c3.metric("Avg Teaching Score", f"{df['rating_1'].mean():.2f}")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Score Distribution")
        st.plotly_chart(px.histogram(df, x='rating_1', nbins=20), use_container_width=True)
    with c2:
        st.subheader("Grading Fairness")


        def cat(s): return "Fair" if s >= 8 else "Average" if s >= 5 else "Strict"


        df['grade_type'] = df['rating_3'].apply(cat)
        counts = df['grade_type'].value_counts().reset_index()
        counts.columns = ['Type', 'Count']
        st.plotly_chart(px.pie(counts, values='Count', names='Type'), use_container_width=True)

    st.divider()
    st.subheader("AI Professor Segmentation")
    st.info("Professors grouped by teaching style (K-Means Clustering).")

    fig_cluster = px.scatter(
        prof_profile,
        x='rating_3', y='rating_1',
        color='Cluster Name', size='comment_count',
        hover_name=prof_profile.index,
        title="Clusters: Fairness (X) vs. Quality (Y)",
        labels={'rating_3': 'Fairness (Strict <-> Fair)', 'rating_1': 'Quality Score'},
        height=500
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.divider()
    st.subheader("Correlation Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if 'has_homework' in prof_profile.columns:
            fig_hw = px.scatter(prof_profile, x='has_homework', y='rating_1', trendline="ols",
                                title="Homework vs Score")
            st.plotly_chart(fig_hw, use_container_width=True)
    with col2:
        if 'has_attendance' in prof_profile.columns:
            fig_att = px.scatter(prof_profile, x='has_attendance', y='rating_1', trendline="ols",
                                 title="Attendance vs Score")
            st.plotly_chart(fig_att, use_container_width=True)

    st.divider()
    st.subheader("Outlier Detection")

    gems, polarizing = get_outliers(prof_profile)

    c1, c2 = st.columns(2)
    with c1:
        st.success("**Underrated Gems**")
        st.caption("High Score (>9.0) but Low Reviews (<=15)")
        if not gems.empty:
            st.table(gems[['rating_1', 'department']].rename(columns={'rating_1': 'Score'}))
        else:
            st.write("No gems found.")

    with c2:
        st.warning("**High Risk / Polarizing**")
        st.caption("Strict Attendance + Unfair Grading + Good Teaching")
        if not polarizing.empty:
            st.table(polarizing[['rating_1', 'rating_3', 'has_attendance']])
        else:
            st.write("No extreme outliers found.")

# ---------------------------------------------------------
# PAGE 2: SEARCH & FILTER (UPDATED)
# ---------------------------------------------------------
elif page == "Search & Filter":
    st.title("Search Professors")
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("Search Name")
    with col_sort:
        sort_opt = st.selectbox("Sort By", ["Highest Score", "Most Comments"])

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Options")

    # 1. Basic Filters
    min_score = st.sidebar.slider("Min Score", 0.0, 10.0, 5.0)

    # 2. Dept & Course Filters (NEW!)
    st.sidebar.subheader("Department & Course")

    # Get unique departments safely
    all_depts = sorted(prof_profile['department'].unique().tolist()) if 'department' in prof_profile.columns else []
    dept_filter = st.sidebar.selectbox("Department", ["All"] + all_depts)

    course_filter = st.sidebar.text_input("Course Name", placeholder="e.g. فیزیک")

    # 3. Style Filters
    st.sidebar.subheader("Teaching Style")
    p_ui = st.sidebar.radio("Project?", ["Any", "Yes", "No"])
    p_arg = True if p_ui == "Yes" else False if p_ui == "No" else None

    h_arg = False if st.sidebar.checkbox("Avoid Heavy Homework?", False) else None
    a_arg = False if st.sidebar.checkbox("Avoid Strict Attendance?", False) else None

    # --- CALL BACKEND ---
    results = get_hybrid_recommendation(
        prof_profile, sim_df,
        target_prof=None,
        min_score=min_score,
        project_based=p_arg,
        heavy_homework=h_arg,
        strict_attendance=a_arg,
        department=dept_filter,  # <--- Pass Dept
        course_name=course_filter,  # <--- Pass Course
        top_n=1000
    )

    # --- DISPLAY RESULTS ---
    if results is not None and not results.empty:
        # Local Text Search (Name Only)
        if search_query: results = results[results.index.str.contains(search_query, case=False)]

        # Sorting
        if sort_opt == "Highest Score":
            results = results.sort_values(by="rating_1", ascending=False)
        else:
            results = results.sort_values(by="comment_count", ascending=False)

        st.subheader(f"Found {len(results)} Professors")

        # Select columns to display
        cols = ['rating_1', 'rating_3', 'department', 'lesson_name', 'has_project', 'has_attendance']
        cols = [c for c in cols if c in results.columns]

        st.dataframe(results[cols], use_container_width=True)
    else:
        st.warning("No professors found matching your filters.")

# ---------------------------------------------------------
# PAGE 3: PROFILE
# ---------------------------------------------------------
elif page == "Professor Profile":
    st.title("Professor Profile")
    prof = st.selectbox("Select Professor", sorted(prof_profile.index))
    if prof:
        data = prof_profile.loc[prof]
        comments = df[df['professor_name'] == prof]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{data['rating_1']:.1f}")
        c2.metric("Reviews", f"{len(comments)}")
        c3.metric("Project", f"{data['has_project']:.2f}")
        c4.metric("Attendance", f"{data['has_attendance']:.2f}")

        st.info(f"Department: {data.get('department', 'N/A')} | Courses: {data.get('lesson_name', 'N/A')}")
        st.divider()

        c_radar, c_txt = st.columns(2)
        with c_radar:
            vals = data[['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6']].tolist();
            vals += vals[:1]
            cols = ['Quality', 'Behavior', 'Fairness', 'Knowledge', 'Comm.', 'Punctuality'];
            cols += cols[:1]
            st.plotly_chart(go.Figure(go.Scatterpolar(r=vals, theta=cols, fill='toself')), use_container_width=True)
        with c_txt:
            st.subheader("Recent Comments")
            for i, row in comments.head(3).iterrows():
                st.info(f"\"{row['clean_comment_text'][:150]}...\"")

# ---------------------------------------------------------
# PAGE 4: COMPARE
# ---------------------------------------------------------
elif page == "Compare":
    st.title("Compare Professors")
    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Professor A", prof_profile.index, key='p1')
    with c2:
        p2 = st.selectbox("Professor B", prof_profile.index, index=1, key='p2')

    if p1 and p2:
        d1, d2 = prof_profile.loc[p1], prof_profile.loc[p2]
        comp = pd.DataFrame({
            'Metric': ['Score', 'Fairness', 'Project', 'Homework', 'Attendance'],
            p1: [f"{d1['rating_1']:.1f}", f"{d1['rating_3']:.1f}", f"{d1['has_project']:.2f}",
                 f"{d1['has_homework']:.2f}", f"{d1['has_attendance']:.2f}"],
            p2: [f"{d2['rating_1']:.1f}", f"{d2['rating_3']:.1f}", f"{d2['has_project']:.2f}",
                 f"{d2['has_homework']:.2f}", f"{d2['has_attendance']:.2f}"]
        }).set_index('Metric')
        st.table(comp)

        fig = go.Figure()
        cols = ['Quality', 'Behavior', 'Fairness', 'Knowledge', 'Comm.', 'Punctuality'];
        cols += cols[:1]
        v1 = d1[['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6']].tolist();
        v1 += v1[:1]
        fig.add_trace(go.Scatterpolar(r=v1, theta=cols, fill='toself', name=p1))
        v2 = d2[['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5', 'rating_6']].tolist();
        v2 += v2[:1]
        fig.add_trace(go.Scatterpolar(r=v2, theta=cols, fill='toself', name=p2))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: RECOMMENDER
# ---------------------------------------------------------
elif page == "Recommender":
    st.title("AI Recommender")
    st.markdown("Find the perfect match for your learning style.")

    with st.form("rec_form"):
        # Row 1: Target Prof + Score
        c1, c2 = st.columns(2)
        target = c1.selectbox("Similar to (Optional)", ["None"] + sorted(prof_profile.index.tolist()))
        min_r = c2.slider("Min Score", 0.0, 10.0, 7.0)

        # Row 2: Department + Course
        c3, c4 = st.columns(2)
        all_depts = sorted(prof_profile['department'].unique().tolist()) if 'department' in prof_profile.columns else []
        dept_sel = c3.selectbox("Department", ["All"] + all_depts)
        course_txt = c4.text_input("Course Name (e.g. 'فیزیک')")

        # Row 3: Preferences
        c5, c6, c7 = st.columns(3)
        p_pref = c5.selectbox("Project", ["Any", "Yes", "No"])
        h_pref = c6.selectbox("Homework", ["Any", "Heavy OK", "Light Only"])
        a_pref = c7.selectbox("Attendance", ["Any", "Strict OK", "Chill Only"])

        if st.form_submit_button("Find Match"):
            t_arg = None if target == "None" else target
            p_arg = True if p_pref == "Yes" else False if p_pref == "No" else None
            h_arg = False if h_pref == "Light Only" else True if h_pref == "Heavy OK" else None
            a_arg = False if a_pref == "Chill Only" else True if a_pref == "Strict OK" else None

            recs = get_hybrid_recommendation(
                prof_profile, sim_df, target_prof=t_arg, min_score=min_r,
                project_based=p_arg, heavy_homework=h_arg, strict_attendance=a_arg,
                department=dept_sel, course_name=course_txt
            )

            if recs is not None and not recs.empty:
                st.success(f"Found {len(recs)} matches!")
                for name, row in recs.iterrows():
                    with st.expander(f"🏆 {name} (Score: {row['rating_1']:.1f})", expanded=True):
                        st.write(
                            f"**Dept:** {row.get('department', 'N/A')} | **Courses:** {row.get('lesson_name', 'N/A')}")
                        st.write(
                            f"Fairness: {row['rating_3']:.1f} | Project: {row['has_project']:.2f} | Attendance: {row['has_attendance']:.2f}")
            else:
                st.error("No matches found.")
