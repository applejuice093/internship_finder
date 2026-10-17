import os
import streamlit as st
from google import genai
from google.genai import types

from core import (
    OFFLINE_QUESTIONS,
    build_question_prompt,
    extract_skills_from_text,
    load_internships,
    parse_questions,
    rag_semantic_search,
    SUPPORTED_RESUME_TYPES,
    read_resume_text,
)



def get_gemini_api_key():
    """Read the Gemini API key from the environment or .streamlit/secrets.toml.

    Never hard-code the key in this file: the repository is public.
    Returns an empty string when no key is configured (offline mode).
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # no secrets.toml present
        return ""


GEMINI_API_KEY = get_gemini_api_key()
# Override with e.g. GEMINI_MODEL=gemini-2.5-pro when the default is retired.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


st.set_page_config(page_title="PM Internship Validator", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #1e293b; font-weight: 800 !important; tracking: -0.02em; }
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
        padding: 24px !important;
        transition: all 0.3s ease;
    }
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; 
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border: none !important;
    }
    .skill-badge { background-color: #e0f2fe; color: #0369a1; padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85em; display: inline-block; margin: 4px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_internship_database(url):
    """Load internship postings, cached for 10 minutes.

    Set INTERNSHIP_DB_URL to a JSON array (e.g. a GitHub Gist "Raw" URL)
    shaped like core.BUILTIN_INTERNSHIPS to fetch live postings. Without
    it, or if the download fails, the built-in sample postings are used.
    """
    return load_internships(url)

def add_notice(message):
    """Queue a warning for the quiz page.

    st.warning() called just before st.rerun() is wiped by the rerun, so
    messages produced while preparing the quiz are kept in session state.
    """
    st.session_state.notices.append(message)


def generate_dynamic_questions(skills, api_key, num_questions=3):
    if not api_key:
        add_notice("Running in Offline Mode: Using built-in question bank.")
        return OFFLINE_QUESTIONS[:num_questions]

    try:
        client = genai.Client(api_key=api_key)
        prompt = build_question_prompt(skills, num_questions)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            # Ask for raw JSON so the reply does not come wrapped in markdown.
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        return parse_questions(response.text)
    except Exception as e:  # network, quota, bad key or unusable response
        add_notice(f"AI question generation failed ({e}). Using the built-in question bank instead.")
        return OFFLINE_QUESTIONS[:num_questions]

def read_file_smart(uploaded_file):
    return read_resume_text(uploaded_file.name, uploaded_file.getvalue())

if 'app_step' not in st.session_state: st.session_state.app_step = 'upload' 
if 'parsed_data' not in st.session_state: st.session_state.parsed_data = {}
if 'quiz_state' not in st.session_state: st.session_state.quiz_state = {'questions': [], 'current_q': 0, 'score': 0}
if 'notices' not in st.session_state: st.session_state.notices = []

if st.session_state.app_step == 'upload':
    st.markdown("<h1 style='text-align: center;'>AI-Based Internship Recommendation Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1em;'>Dynamic profile validation and semantic matching for the PM Internship Scheme.</p>", unsafe_allow_html=True)
    
    st.write("")
    
    with st.container(border=True):
        st.markdown("### 👤 Candidate Profile")
        col1, col2 = st.columns(2)
        with col1: user_edu = st.selectbox("Highest Education", ["12th Pass", "Diploma", "Graduate"])
        with col2: user_loc = st.selectbox("Preferred Location", ["Remote", "Maharashtra", "Andhra Pradesh", "Delhi"])
        
        st.markdown("---")
        st.markdown("### 📄 Resume Data")
        uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=SUPPORTED_RESUME_TYPES)
        
        if st.button("Initialize Deep Analysis & Generate Assessment", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Parsing documents & fetching dynamic assessment via internet..."):
                    try:
                        text = read_file_smart(uploaded_file)
                    except Exception:  # corrupt, encrypted or mislabelled file
                        st.error("Could not read this file. Please upload a valid, unencrypted PDF or DOCX resume.")
                        st.stop()

                    st.session_state.notices = []
                    if not text.strip():
                        add_notice("No text could be extracted from the resume (is it a scanned image?). "
                                   "The assessment will use general questions.")

                    skills = extract_skills_from_text(text)
                    if not skills:
                        skills = ["general computer skills"]

                    st.session_state.parsed_data = {
                        "skills": skills,
                        "location": user_loc,
                        "education": user_edu,
                    }
                
                    st.session_state.quiz_state['questions'] = generate_dynamic_questions(skills, GEMINI_API_KEY)
                    st.session_state.app_step = 'quiz'
                    st.rerun()
            else:
                st.error("Please upload a resume first.")

elif st.session_state.app_step == 'quiz':
    st.markdown("<h1>🧠 Dynamic Skill Assessment</h1>", unsafe_allow_html=True)
    st.caption("These questions were generated in real-time based on the skills found in your resume.")
    for notice in st.session_state.notices:
        st.warning(notice)
    
    qs = st.session_state.quiz_state['questions']
    
    if not qs:
        st.error("Failed to load questions. Returning to home.")
        if st.button("Go Back"):
            st.session_state.app_step = 'upload'
            st.rerun()
    else:
        idx = st.session_state.quiz_state['current_q']
        progress = (idx) / len(qs)
        st.progress(progress, text=f"Evaluating Skill Confidence: Question {idx + 1} of {len(qs)}")
        
        with st.container(border=True):
            current_q = qs[idx]
            st.markdown(f"### {current_q['q']}")
            
            choice = st.radio("Select your answer:", current_q['options'], index=None, key=f"q_{idx}")
            
            if st.button("Submit Answer", type="primary"):
                if choice is None:
                    # Without this, clicking Submit by accident skipped the question as wrong.
                    st.warning("Please select an answer before submitting.")
                    st.stop()
                if choice == current_q['a']:
                    st.session_state.quiz_state['score'] += 1
                
                st.session_state.quiz_state['current_q'] += 1
                
                if st.session_state.quiz_state['current_q'] >= len(qs):
                    st.session_state.app_step = 'results'
                st.rerun()

elif st.session_state.app_step == 'results':
    data = st.session_state.parsed_data
    quiz = st.session_state.quiz_state
    
    st.markdown("<h1>🎯 PM Internship Match Dashboard</h1>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Validation Score", f"{quiz['score']} / {len(quiz['questions'])}", delta="Verified")
    c2.metric("Extracted Skills", len(data['skills']))
    c3.metric("Profile Status", "Ready for Matching")
    
    st.markdown("### 🔑 Verified Competencies")
    skills_html = "".join([f"<span class='skill-badge'>{skill.upper()}</span>" for skill in data['skills']])
    st.markdown(skills_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💼 Semantic AI Recommendations")
    
    with st.spinner("Running TF-IDF vectorization & cosine similarity against the internship database..."):
        db_jobs, db_error = fetch_internship_database(os.environ.get("INTERNSHIP_DB_URL", ""))
        if db_error:
            st.warning(db_error + " Showing the built-in sample internships instead.")
        top_matches = rag_semantic_search({"location": data["location"], "education": data["education"]}, data["skills"], db_jobs, top_n=3)
    
    for intern in top_matches:
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"### {intern['title']} @ {intern['company']}")
                st.caption(f"📍 {intern['location']} | 🎓 {intern['education']}")
                st.write(intern['description'])
            with col_b:
                color = "#059669" if intern['score'] > 60 else "#d97706"
                st.markdown(f"<h2 style='color: {color}; text-align: right;'>{intern['score']}% Match</h2>", unsafe_allow_html=True)
                st.link_button("Apply on MCA Portal ↗", url=intern['apply_url'], use_container_width=True)

    st.write("")
    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.session_state.app_step = 'upload'
        st.session_state.parsed_data = {}
        st.session_state.quiz_state = {'questions': [], 'current_q': 0, 'score': 0}
        st.session_state.notices = []
        st.rerun()