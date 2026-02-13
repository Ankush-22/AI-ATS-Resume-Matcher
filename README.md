# 🚀 Smart ATS Optimizer: Resume–Job Description Matcher

An industry-agnostic recruitment tool that leverages **Deep Learning** to bridge the gap between job seekers and employers. Unlike traditional ATS systems that rely on rigid keyword matching, this tool uses **Natural Language Processing (NLP)** to understand the semantic context of a candidate's experience.

## 🧠 The Engine
- **Model:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Math:** Cosine Similarity for high-dimensional vector alignment.
- **Extraction:** Frequency-based N-gram analysis for dynamic skill discovery.

## ✨ Key Features
- **Semantic Match Scoring:** Measures the "thematic fit" of a resume against any Job Description.
- **Dynamic Skill Gap Analysis:** Automatically identifies missing technical competencies across various domains (Data Science, DevOps, Blockchain, etc.).
- **Interactive Visualizations:** Integrated **Plotly Radar Charts** to provide a 360-degree view of candidate-job alignment.
- **Universal File Support:** Robust text extraction for `.pdf` and `.docx` formats.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Streamlit, Sentence-Transformers, Scikit-learn, Plotly, PyPDF2, python-docx
- **Deployment:** Streamlit Cloud

## 📈 Methodology
1. **Text Extraction:** Raw data is pulled from uploaded resumes and job descriptions.
2. **Vectorization:** Text is converted into 384-dimensional dense vectors.
3. **Similarity Calculation:** Cosine similarity determines the semantic overlap.
4. **Gap Analysis:** A specialized taxonomy engine filters and identifies missing technical keywords.

## 🚀 Installation & Usage

Run the following commands in order:

```bash
git clone https://github.com/Ankush-22/AI-ATS-Resume-Matcher.git
cd AI-ATS-Resume-Matcher
python -m venv ats_env
source ats_env/bin/activate   # Windows: ats_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py