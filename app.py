import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# GANVORAX AI | Multi-page Streamlit application
# Keeps the original TF-IDF + cosine-similarity recommendation
# approach and the original six sample job records.
# ------------------------------------------------------------

st.set_page_config(
    page_title="GANVORAX AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained, responsive visual theme.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.22);}
      .brand {font-size: 1.8rem; font-weight: 800; letter-spacing: .02em;}
      .muted {color: #8b93a7; font-size: .95rem;}
      .hero {
        padding: 2rem; border-radius: 18px;
        background: linear-gradient(125deg, rgba(37,99,235,.18), rgba(16,185,129,.12));
        border: 1px solid rgba(128,128,128,.22);
      }
      .small-note {font-size: .82rem; color: #9299a8;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        padding: 14px; border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Original sample dataset
jobs = {
    "Job_ID": [101, 102, 103, 104, 105, 106],
    "Job_Title": [
        "Python Developer",
        "Data Scientist",
        "Machine Learning Engineer",
        "Java Developer",
        "Data Analyst",
        "AI Engineer",
    ],
    "Company": [
        "TechNova",
        "DataWorld",
        "AI Solutions",
        "CodeCraft",
        "Analytics Hub",
        "FutureAI",
    ],
    "Skills": [
        "Python SQL Django API",
        "Python Machine Learning Statistics SQL",
        "Python Machine Learning Deep Learning TensorFlow",
        "Java Spring Boot SQL OOP",
        "Python SQL Excel Power BI Statistics",
        "Python Deep Learning NLP TensorFlow",
    ],
}
df = pd.DataFrame(jobs)
df["Job_Text"] = df["Job_Title"] + " " + df["Skills"]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["Cleaned_Text"] = df["Job_Text"].apply(clean_text)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["Cleaned_Text"])

skill_database = [
    "Python", "SQL", "Django", "API", "Machine Learning", "Statistics",
    "Deep Learning", "TensorFlow", "Java", "Spring Boot", "Excel",
    "Power BI", "NLP", "OOP",
]

def extract_skills(text: str):
    found = []
    lower_text = text.lower()
    for skill in skill_database:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, lower_text):
            found.append(skill)
    return found

def recommend_jobs(resume_text: str) -> pd.DataFrame:
    resume_vector = tfidf.transform([clean_text(resume_text)])
    scores = cosine_similarity(resume_vector, tfidf_matrix).flatten()
    results = df[["Job_ID", "Job_Title", "Company", "Skills"]].copy()
    results["Match Score"] = scores * 100
    return results.sort_values("Match Score", ascending=False).reset_index(drop=True)

# Persist analysis when users navigate between pages.
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""

# Sidebar navigation
with st.sidebar:
    st.markdown('<div class="brand">💼 GANVORAX AI</div>', unsafe_allow_html=True)
    st.caption("Intelligent Job Matching Platform")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Resume Analyzer",
            "Job Recommendations",
            "Skill Gap Dashboard",
            "About",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("PROJECT STATUS")
    if st.session_state.results is not None:
        st.success("Resume analysed")
    else:
        st.info("Awaiting resume analysis")
    st.caption("GANVORAX AI · ML & Data Science")

# HOME
if page == "Home":
    st.markdown(
        """
        <div class="hero">
          <div class="brand">Welcome to GANVORAX AI</div>
          <h2>Find opportunities that match your skills.</h2>
          <p class="muted">An intelligent job matching and recruitment recommendation system
          powered by TF-IDF text features and cosine similarity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("Your career toolkit")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📄 Resume Analyzer")
        st.write("Upload a PDF resume or paste your resume information for analysis.")
        st.markdown("### 🎯 Job Recommendations")
        st.write("See the sample job roles ranked by textual similarity to your resume.")
    with c2:
        st.markdown("### 📊 Skill Gap Dashboard")
        st.write("Compare recognized skills with the skills required for a selected role.")
        st.markdown("### 🧠 Matching technology")
        st.write("TF-IDF vectorization and cosine similarity power the recommendation score.")
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Sample job roles", len(df))
    m2.metric("Recognized skills", len(skill_database))
    m3.metric("Analysis method", "TF-IDF")
    st.info("To begin, choose **Resume Analyzer** from the left navigation.")

# RESUME ANALYZER
elif page == "Resume Analyzer":
    st.title("📄 Resume Analyzer")
    st.write("Upload a text-based PDF resume or paste your resume information below.")
    uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"], key="resume_pdf")
    extracted_text = ""
    if uploaded_file is not None:
        try:
            reader = PdfReader(uploaded_file)
            extracted_text = " ".join(page.extract_text() or "" for page in reader.pages)
            if extracted_text.strip():
                st.success(f"Resume text extracted successfully: {uploaded_file.name}")
                st.session_state.uploaded_name = uploaded_file.name
            else:
                st.warning("No selectable text was found. This may be a scanned PDF. Please paste the resume text below.")
        except Exception as exc:
            st.error(f"Unable to read this PDF: {exc}")

    existing_text = extracted_text or st.session_state.resume_text
    manual_resume = st.text_area(
        "Resume information",
        value=existing_text,
        height=220,
        placeholder="Enter your skills, education and experience...",
        key="resume_text_input",
    )

    if st.button("Analyze Resume & Find Jobs", type="primary", use_container_width=True):
        if not manual_resume.strip():
            st.warning("Please upload a text-based PDF or enter your resume information.")
        else:
            st.session_state.resume_text = manual_resume
            st.session_state.results = recommend_jobs(manual_resume)
            st.success("Analysis complete! Open Job Recommendations or Skill Gap Dashboard from the sidebar.")

    if st.session_state.results is not None:
        st.divider()
        st.subheader("Analysis status")
        a, b = st.columns(2)
        a.metric("Resume characters", len(st.session_state.resume_text))
        b.metric("Recognized skills", len(extract_skills(st.session_state.resume_text)))
        st.caption("Your analysis is retained while you navigate within this app session.")

# JOB RECOMMENDATIONS
elif page == "Job Recommendations":
    st.title("🎯 Job Recommendations")
    st.write("Your top matches are ranked using text similarity against the sample job dataset.")
    if st.session_state.results is None:
        st.warning("No resume has been analysed yet. Visit Resume Analyzer and run the analysis first.")
        st.button("Go to Resume Analyzer", on_click=lambda: st.session_state.update())
    else:
        results = st.session_state.results
        top_jobs = results.head(3)
        st.subheader("Top 3 recommended roles")
        st.dataframe(
            top_jobs[["Job_Title", "Company", "Match Score"]].style.format({"Match Score": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Matching score visualization")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(top_jobs["Job_Title"], top_jobs["Match Score"])
        ax.set_xlabel("Text similarity score (%)")
        ax.set_title("GANVORAX AI — Top Job Matches")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("View all sample job matches"):
            st.dataframe(
                results[["Job_Title", "Company", "Skills", "Match Score"]].style.format({"Match Score": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("Match scores indicate textual similarity to this sample dataset, not the probability of being hired.")

# SKILL GAP DASHBOARD
elif page == "Skill Gap Dashboard":
    st.title("📊 Skill Gap Dashboard")
    st.write("Compare the skills recognized in your resume with the skills listed for a selected role.")
    if st.session_state.results is None:
        st.warning("Please analyse a resume first using the Resume Analyzer page.")
    else:
        selected_job = st.selectbox("Select a job for skill analysis", df["Job_Title"].tolist())
        job = df.loc[df["Job_Title"] == selected_job].iloc[0]
        candidate_skills = extract_skills(st.session_state.resume_text)
        required_skills = extract_skills(job["Skills"])
        matched = [skill for skill in required_skills if skill in candidate_skills]
        missing = [skill for skill in required_skills if skill not in candidate_skills]
        coverage = len(matched) / len(required_skills) * 100 if required_skills else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Required skills", len(required_skills))
        c2.metric("Matched skills", len(matched))
        c3.metric("Skill coverage", f"{coverage:.1f}%")

        st.subheader("Skill breakdown")
        left, right = st.columns(2)
        with left:
            st.markdown("**Matched skills**")
            if matched:
                st.success(" · ".join(matched))
            else:
                st.write("No matched skills found.")
        with right:
            st.markdown("**Missing skills**")
            if missing:
                st.warning(" · ".join(missing))
            else:
                st.success("No missing skills found.")
        if required_skills:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.barh(["Skill coverage"], [coverage])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Coverage (%)")
            ax.set_title(f"Skill coverage for {selected_job}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Skill coverage is based on the application's built-in skill list and sample job requirements.")

# ABOUT
elif page == "About":
    st.title("ℹ️ About GANVORAX AI")
    st.markdown(
        """
        **GANVORAX AI** is a machine-learning project for job matching and recruitment
        recommendation. It compares resume text with job title and skill descriptions.

        **Core workflow**
        1. Extract text from a PDF resume or accept pasted resume text.
        2. Clean and normalize text.
        3. Convert job descriptions into TF-IDF vectors.
        4. Compare the resume vector with job vectors using cosine similarity.
        5. Rank job roles and compare recognized skills.
        """
    )
    st.subheader("Technology stack")
    st.write("Python · Streamlit · Pandas · Scikit-learn · TF-IDF · Cosine Similarity · PyPDF2 · Matplotlib")
    st.warning(
        "This version uses six illustrative sample job records, not live vacancies. "
        "Similarity scores are not hiring probabilities. Resume text is processed in the app session; "
        "avoid uploading sensitive personal information to a public demo."
    )

st.divider()
st.caption("© GANVORAX AI | Machine Learning & Data Science Project")
