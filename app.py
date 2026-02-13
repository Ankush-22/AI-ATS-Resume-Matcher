import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import re
import collections
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Resume Advisor Pro", layout="wide")

@st.cache_resource
def load_model():
    # Loading the transformer model for semantic meaning
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf_reader.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def clean_text(text):
    # Basic cleaning: remove extra whitespace and lowercase
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def create_radar_chart(found, missing):
    categories = found + missing
    if not categories or len(categories) < 3: # Radar charts look best with 3+ items
        return None
        
    # Scoring: 10 for skills you have, 2 for what you don't
    scores = [10] * len(found) + [2] * len(missing)
    
    # Close the radar loop
    categories = [*categories, categories[0]]
    scores = [*scores, scores[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Skill Match',
        line=dict(color='#00CC96')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=450,
        title="Visual Skill Gap Analysis"
    )
    return fig

def get_dynamic_skills(text):
    """
    Professional skill extractor: Only identifies words that exist 
    in a pre-defined library of technical and soft skills.
    """
    # 1. The Skill Knowledge Base (Add any specific field terms here)

    skill_library = set([
        # --- Languages & Frameworks ---
        'html', 'html5', 'css', 'css3', 'js', 'javascript', 'jquery', 'angular', 'react', 'vue',
        'web devlopment', 'web disign', 'bootstrap', 'svg', 'photoshop', 'illustrator', 'sketch', 'indesign', 'figma', 'adobe', 
        'wireframes', 'layouts', 'rwd', 'responsive', 'ui', 'ux', 'flash', 'corel draw',
        'python', 'sql', 'machine learning', 'data science', 'tensorflow', 'keras', 'pytorch',
        'nlp', 'pandas', 'numpy', 'scikit learn', 'seaborn', 'matplotlib', 'java', 'spark', 'hadoop', 
        'tableau', 'cpp', 'c++', 'c#', 'spring', 'django', 'Power BI', 'Data Analysis',
        'flask', 'node', 'express', 'golang', 'rust', 'typescript', 'php', 'laravel',

        # --- Cloud & DevOps ---
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'ansible', 
        'terraform', 'ci/cd', 'devops', 'linux', 'bash', 'shell', 'promethus', 'grafana',

        # --- Core CS & Software Engineering ---
        'oop', 'object oriented', 'data structures', 'algorithms', 'dsa', 'dbms', 'os', 
        'operating systems', 'system design', 'microservices', 'rest api', 'graphql', 
        'unit testing', 'selenium', 'postman', 'sdlc', 'agile', 'scrum', 'kanban',

        # --- Databases & Tools ---
        'git', 'github', 'gitlab', 'mysql', 'mongodb', 'postgresql', 'redis', 'oracle', 
        'jira', 'confluence', 'slack', 'trello', 'vscode', 'intellij', 'pycharm',

        # --- Advanced Tech Fields ---
        'blockchain', 'solidity', 'ethereum', 'cybersecurity', 'cryptography', 'penetration testing',
        'cloud security', 'big data', 'data engineering', 'snowflake', 'databricks',

        # --- Business & Soft Skills ---
        'communication', 'problem solving', 'teamwork', 'leadership', 'analytical', 
        'critical thinking', 'project management', 'stakeholder management', 'time management'
    ])

    
    # 2. Extract all words and 2-word phrases (like 'machine learning')
    text_lower = text.lower()
    found_skills = []
    
    # Check for single-word matches
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    for word in words:
        if word in skill_library:
            found_skills.append(word.capitalize())
            
    # Check for specific 2-word phrases
    phrases = [
    # Data & AI
    'machine learning', 'data science', 'deep learning', 'neural networks', 
    'big data', 'data analysis', 'data visualization', 'natural language processing',
    'web devlopment', 'web design',
    
    # Core CS & Software Engineering
    'data structures', 'algorithms', 'object oriented', 'system design', 
    'software development', 'web development', 'full stack', 'distributed systems',
    'microservices architecture', 'rest api', 'unit testing', 'test driven development',
    
    # Cloud & DevOps
    'cloud computing', 'google cloud', 'amazon web services', 'ci/cd pipeline', 
    'infrastructure as code', 'container orchestration',
    
    # Professional & Soft Skills
    'problem solving', 'project management', 'agile methodology', 'critical thinking',
    'time management', 'stakeholder management', 'communication skills'
    ]

    for phrase in phrases:
        if phrase in text_lower:
            found_skills.append(phrase.title())
    
    # 3. Remove duplicates and return
    return sorted(list(set(found_skills)))

# --- UI LAYOUT ---
st.title("🚀 Smart ATS Optimizer")
st.markdown("### Industry-Agnostic Resume Matcher & Skill Gap Analyzer")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📝 Step 1: Paste Job Description")
    jd_input = st.text_area("Works for any field (Blockchain, Cyber, Design, etc.)", height=250)

with col2:
    st.markdown("#### 📤 Step 2: Upload Your Resume")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

# --- DYNAMIC SKILL SETTINGS ---
with st.expander("🔬 AI Dynamic Skill Extraction (Auto-Detected)"):
    if jd_input:
        # Automatically pull keywords from whatever JD is pasted
        auto_detected = get_dynamic_skills(jd_input)
        tech_stack_input = st.text_input("The AI identified these terms to track:", value=", ".join(auto_detected))
    else:
        st.write("Paste a Job Description above to see the AI extract keywords automatically.")
        tech_stack_input = ""

# --- ANALYSIS LOGIC ---
if st.button("Run Deep Analysis"):
    if jd_input and uploaded_file:
        with st.spinner('Analyzing content and calculating semantic similarity...'):
            # 1. Extract and Clean Text
            if uploaded_file.type == "application/pdf":
                resume_raw = extract_text_from_pdf(uploaded_file)
            else:
                resume_raw = extract_text_from_docx(uploaded_file)
            
            clean_jd = clean_text(jd_input)
            clean_res = clean_text(resume_raw)
            
            # 2. Semantic Match Score (The "AI Brain")
            jd_vec = model.encode([clean_jd])
            res_vec = model.encode([clean_res])
            # Cast numpy float32 to python float to avoid Streamlit errors
            score = float(cosine_similarity(jd_vec, res_vec)[0][0])
            
            # 3. Dynamic Skill Gap Analysis
            # Convert user-edited tech_stack_input back to a list
            target_skills = [s.strip().lower() for s in tech_stack_input.split(",")]
            found_in_res = [s.capitalize() for s in target_skills if s in clean_res]
            missing_from_res = [s.capitalize() for s in target_skills if s not in clean_res]

        # --- DISPLAY RESULTS ---
        st.divider()
        
        # Big Metric
        st.header(f"Overall Match Score: {score*100:.1f}%")
        st.progress(score)

        st.divider()
        # We only take the top 12 skills to keep the chart readable
        chart_fig = create_radar_chart(found_in_res[:6], missing_from_res[:6])
        if chart_fig:
            st.plotly_chart(chart_fig, use_container_width=True)
        # ----------------------------------------------

        # Analysis Columns
        res_col1, res_col2, res_col3 = st.columns(3)

        # Analysis Columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.success("### ✅ Found in Resume")
            if found_in_res:
                for s in found_in_res: st.markdown(f"- **{s}**")
            else:
                st.write("No targeted keywords found.")

        with res_col2:
            st.error("### ❌ Missing Skills")
            if missing_from_res:
                for s in missing_from_res: st.markdown(f"- **{s}**")
            else:
                st.write("Perfect! No missing keywords.")

        with res_col3:
            st.info("### 💡 ATS Advice")
            if score < 0.45:
                st.warning("Low Semantic Match. Your resume's overall 'vibe' doesn't match this industry. Consider rewriting your Summary.")
            elif score < 0.65:
                st.write("Partial Match. You have the right background, but you are missing the specific keywords listed in the middle column.")
            else:
                st.balloons()
                st.write("Strong Match! Your resume and this JD speak the same language.")

    else:
        st.error("Please provide both a Job Description and a Resume file.")