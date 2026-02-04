import pandas as pd
import sys
from backend import load_and_prep_data, build_similarity_matrix, get_hybrid_recommendation

# Configuration
DATA_PATH = 'data/processed/sentiment_data.csv'


def print_separator(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def run_tests():
    # ---------------------------------------------------------
    # 1. SETUP & DATA LOADING
    # ---------------------------------------------------------
    print_separator("STEP 1: INITIALIZATION")

    print(f"Loading data from: {DATA_PATH}")
    df = load_and_prep_data(DATA_PATH)

    if df is None:
        print("Error: Could not load data. Check file path.")
        sys.exit(1)

    print(f"Data Loaded. Shape: {df.shape}")

    prof_profile, sim_df = build_similarity_matrix(df)
    print(f"Matrix Built. Shape: {sim_df.shape}")
    # Validates that you have a square matrix (N professors x N professors)
    print(f"Matrix Built. Active Professors: {len(prof_profile)}")
    print(f"One-Hot Encoded Departments: {any('dept_' in col for col in sim_df.index.name or [])}")

    # ---------------------------------------------------------
    # 2. TESTING SEARCH FILTERS (Course & Dept)
    # ---------------------------------------------------------
    print_separator("TEST 2: SEARCH CAPABILITIES")

    # Test A: Course Name Search
    search_course = "فیزیک"
    print(f"Test A: Searching for Course containing '{search_course}'...")
    res_course = get_hybrid_recommendation(
        prof_profile, sim_df,
        course_name=search_course,
        top_n=3
    )
    if res_course is not None and not res_course.empty:
        print(res_course[['rating_1', 'lesson_name', 'department']])
    else:
        print("No results found (Check if 'lesson_name' is populated).")

    # Test B: Department Search
    # We dynamically pick a department that exists to avoid errors
    if 'department' in prof_profile.columns:
        valid_dept = prof_profile['department'].mode()[0]
        print(f"\nTest B: Searching for Department '{valid_dept}'...")
        res_dept = get_hybrid_recommendation(
            prof_profile, sim_df,
            department=valid_dept,
            top_n=3
        )
        print(res_dept[['rating_1', 'department', 'comment_count']])

    # ---------------------------------------------------------
    # 3. TESTING LOGIC FILTERS (Homework/Attendance)
    # ---------------------------------------------------------
    print_separator("TEST 3: LOGIC FILTERS")

    # Test C: Find 'Chill' Professors (No Attendance)
    print("Test C: Filter for Chill Attendance (Strict_Attendance=False)...")
    res_chill = get_hybrid_recommendation(
        prof_profile, sim_df,
        strict_attendance=False,
        top_n=3
    )
    if not res_chill.empty:
        # Verify the 'has_attendance' score is low
        print(res_chill[['rating_1', 'has_attendance']])
    else:
        print("No chill professors found.")

    # ---------------------------------------------------------
    # 4. TESTING SIMILARITY (The AI Part)
    # ---------------------------------------------------------
    print_separator("TEST 4: HYBRID SIMILARITY")

    target_prof = prof_profile.index[0]
    print(f"Target Professor: {target_prof}")
    print(f"Target Dept: {prof_profile.loc[target_prof, 'department']}")

    print(f"\nFinding professors similar to '{target_prof}' who give Projects...")
    res_sim = get_hybrid_recommendation(
        prof_profile, sim_df,
        target_prof=target_prof,
        project_based=True,
        top_n=3
    )

    if res_sim is not None:
        print(res_sim[['rating_1', 'department', 'has_project']])
        print("\nNote: Check if Department matches Target")
    else:
        print("Target professor not found in similarity matrix.")


if __name__ == "__main__":
    run_tests()