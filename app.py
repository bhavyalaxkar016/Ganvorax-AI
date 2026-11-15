import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="GANVORAX AI", page_icon="💼", layout="wide")

st.title("💼 GANVORAX AI")
st.subheader("Intelligent Job Matching & Recruitment Recommendation System")
st.write("Upload your resume or paste your resume text to discover suitable job opportunities.")

# Sample dataset from the original project
jobs = {
    "Job_ID": [101, 102, 103, 104, 105, 106],
    "Job_Title": [
        "Python Developer", "Data Scientist", "Machine Learning Engineer",
        "Java Developer", "Data Analyst", "AI Engineer"
    ],
    "Company": [
        "TechNova", "DataWorld", "AI Solutions",
        "CodeCraft", "Analytics Hub", "FutureAI"
    ],
    "Skills": [
        "Python SQL Django API",
        "Python Machine Learning Statistics SQL",
        "Python Machine Learning Deep Learning TensorFlow",
        "Java Spring Boot SQL OOP",
        "Python SQL Excel Power BI Statistics",
        "Python Deep Learning NLP TensorFlow"
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
    "Power BI", "NLP", "OOP"
]

def extract_skills(text: str):
    found = []
    lower_text = text.lower()
    for skill in skill_database:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, lower_text):
            found.append(skill)
    return found

def recommend_jobs(resume_text: str):
    resume_vector = tfidf.transform([clean_text(resume_text)])
    scores = cosine_similarity(resume_vector, tfidf_matrix).flatten()
    results = df[["Job_ID", "Job_Title", "Company", "Skills"]].copy()
    results["Match Score"] = scores * 100
    return results.sort_values("Match Score", ascending=False).reset_index(drop=True)

st.header("1. Upload your resume")
uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
resume_text = ""

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        resume_text = " ".join(page.extract_text() or "" for page in reader.pages)
        if resume_text.strip():
            st.success("Resume uploaded and text extracted successfully.")
        else:
            st.warning("No selectable text was found. If this is a scanned PDF, paste your resume text below.")
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")

manual_resume = st.text_area(
    "Or paste your resume information",
    height=180,
    placeholder="Enter your skills, education and experience..."
)
if manual_resume.strip():
    resume_text = manual_resume

if st.button("Analyze Resume & Find Jobs", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.warning("Please upload a text-based PDF or paste your resume information.")
    else:
        results = recommend_jobs(resume_text)
        top_jobs = results.head(3)

        st.header("2. Top job recommendations")
        st.dataframe(
            top_jobs[["Job_Title", "Company", "Match Score"]].style.format({"Match Score": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )

        st.header("3. Job matching visualization")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(top_jobs["Job_Title"], top_jobs["Match Score"])
        ax.set_xlabel("Text similarity score (%)")
        ax.set_title("Top job matching results")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close(fig)

        st.header("4. Skill gap analysis")
        selected_job = st.selectbox("Select a job for skill analysis", df["Job_Title"].tolist())
        job = df.loc[df["Job_Title"] == selected_job].iloc[0]
        candidate_skills = extract_skills(resume_text)
        required_skills = extract_skills(job["Skills"])
        matched = [skill for skill in required_skills if skill in candidate_skills]
        missing = [skill for skill in required_skills if skill not in candidate_skills]
        coverage = len(matched) / len(required_skills) * 100 if required_skills else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Required skills", len(required_skills))
        c2.metric("Matched skills", len(matched))
        c3.metric("Skill coverage", f"{coverage:.1f}%")

        st.write("**Recognized skills:**", ", ".join(candidate_skills) or "None found")
        st.write("**Matched skills:**", ", ".join(matched) or "None")
        st.write("**Missing skills:**", ", ".join(missing) or "None")

        st.caption(
            "Match scores represent textual similarity to this sample job dataset; "
            "they are not the probability of being hired."
        )

st.divider()
st.caption("GANVORAX AI | Machine Learning & Data Science Project")
