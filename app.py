import streamlit as st
import hashlib
import sqlite3
import re
import io
import pickle
import json
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pypdf
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode('utf-8')

def extract_text(file_bytes: bytes, file_type: str) -> str:
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file_bytes)
    elif file_type == "text/plain":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class StylometricFeatures:
    sentence_length_mean: float
    sentence_length_std: float
    lexical_diversity: float
    function_word_ratio: float
    hedging_density: float
    transition_density: float
    punctuation_diversity: float
    repetition_rate: float
    entropy: float
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StylometricFeatures':
        data = json.loads(json_str)
        return cls(**data)

# ============================================================================
# AI DETECTION ENGINE
# ============================================================================

class AIDetector:
    def __init__(self):
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
        }
        self.hedging_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'would', 'may',
            'suggests', 'indicates', 'appears', 'seems', 'likely'
        }
    
    def calculate_lexical_diversity(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def calculate_function_word_ratio(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        function_count = sum(1 for w in words if w in self.function_words)
        return function_count / len(words)
    
    def calculate_hedging_density(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        hedging_count = sum(1 for w in words if w in self.hedging_words)
        return hedging_count / len(words)
    
    def calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        chars = list(text.lower())
        char_counts = Counter(chars)
        probs = [count/len(chars) for count in char_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        max_entropy = np.log(len(char_counts)) if len(char_counts) > 0 else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def calculate_perplexity(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 5:
            return 0.5
        unique_ratio = len(set(words)) / len(words)
        return 1 - min(unique_ratio, 0.8) / 0.8
    
    def calculate_burstiness(self, text: str) -> float:
        sentences = re.split(r'[.!?]+', text)
        sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        if len(sent_lengths) < 2:
            return 0.5
        cv = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0
        return 1 - min(cv, 0.5) / 0.5
    
    def extract_features(self, text: str) -> StylometricFeatures:
        sentences = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        sent_lengths = [len(s.split()) for s in sentences]
        lexical_div = self.calculate_lexical_diversity(text)
        
        return StylometricFeatures(
            sentence_length_mean=np.mean(sent_lengths) if sent_lengths else 0,
            sentence_length_std=np.std(sent_lengths) if sent_lengths else 0,
            lexical_diversity=lexical_div,
            function_word_ratio=self.calculate_function_word_ratio(text),
            hedging_density=self.calculate_hedging_density(text),
            transition_density=self.calculate_function_word_ratio(text),
            punctuation_diversity=0.5,
            repetition_rate=1 - lexical_div,
            entropy=self.calculate_entropy(text)
        )
    
    def detect(self, text: str) -> Tuple[float, List[str], Dict]:
        features = self.extract_features(text)
        explanations = []
        
        signals = {
            'lexical_diversity': features.lexical_diversity,
            'function_word_ratio': features.function_word_ratio,
            'hedging_density': features.hedging_density,
            'entropy': features.entropy,
            'burstiness': self.calculate_burstiness(text),
            'perplexity': self.calculate_perplexity(text)
        }
        
        normalized = {
            'lexical_diversity': 1 - signals['lexical_diversity'],
            'function_word_ratio': signals['function_word_ratio'],
            'hedging_density': 1 - signals['hedging_density'],
            'entropy': 1 - signals['entropy'],
            'burstiness': signals['burstiness'],
            'perplexity': signals['perplexity']
        }
        
        weights = {'lexical_diversity': 0.15, 'function_word_ratio': 0.20, 
                   'hedging_density': 0.20, 'entropy': 0.15, 'burstiness': 0.15, 'perplexity': 0.15}
        
        ai_score = sum(normalized[k] * weights[k] for k in weights)
        
        if normalized['lexical_diversity'] > 0.7:
            explanations.append("Low lexical diversity (repetitive vocabulary)")
        if normalized['function_word_ratio'] > 0.6:
            explanations.append("High function word density (formal pattern)")
        if normalized['hedging_density'] < 0.3:
            explanations.append("Low hedging language (confident assertions)")
        if normalized['entropy'] < 0.4:
            explanations.append("Low character entropy (predictable text)")
        if normalized['burstiness'] < 0.3:
            explanations.append("Uniform sentence length (AI pattern)")
        if normalized['perplexity'] < 0.3:
            explanations.append("Low perplexity (predictable words)")
        
        metrics = {
            'AI Score': ai_score,
            'Lexical Diversity': features.lexical_diversity,
            'Function Words': features.function_word_ratio,
            'Hedging Density': features.hedging_density,
            'Entropy': features.entropy,
            'Burstiness': signals['burstiness'],
            'Perplexity': signals['perplexity']
        }
        
        return ai_score, explanations, metrics

# ============================================================================
# PLAGIARISM ENGINE
# ============================================================================

class PlagiarismEngine:
    def __init__(self, db_path: str = "integrity.db"):
        self.db_path = db_path
        self.init_database()
        self.model = None
        self.semantic_index = None
        self.chunk_texts = []
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
            self.load_indexes()
        except Exception as e:
            st.warning(f"Model loading issue: {e}")
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            submission_type TEXT,
            text_hash TEXT UNIQUE,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()
    
    def load_indexes(self):
        try:
            self.semantic_index = faiss.read_index("semantic_index.faiss")
            with open("chunk_texts.pkl", "rb") as f:
                self.chunk_texts = pickle.load(f)
        except:
            self.semantic_index = faiss.IndexFlatL2(self.dimension)
            self.chunk_texts = []
    
    def save_indexes(self):
        if self.semantic_index:
            faiss.write_index(self.semantic_index, "semantic_index.faiss")
            with open("chunk_texts.pkl", "wb") as f:
                pickle.dump(self.chunk_texts, f)
    
    def add_submission(self, student_id: str, sub_type: str, text: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        detector = AIDetector()
        features = detector.extract_features(text)
        
        cursor.execute('INSERT INTO submissions (student_id, submission_type, text_hash, features) VALUES (?, ?, ?, ?)',
                      (student_id, sub_type, text_hash, features.to_json()))
        conn.commit()
        conn.close()
        
        if self.model:
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 30:
                    embedding = self.model.encode([sent])[0]
                    self.chunk_texts.append(sent)
                    self.semantic_index.add(np.array([embedding]).astype('float32'))
            self.save_indexes()
    
    def check(self, text: str) -> Tuple[float, List[str], Dict]:
        if not self.model:
            return 0.0, ["Model not available"], {'matches': []}
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        if not sentences:
            return 0.0, ["No substantial sentences"], {'matches': []}
        
        matches = []
        for sent in sentences[:30]:
            if self.semantic_index and self.semantic_index.ntotal > 0:
                embedding = self.model.encode([sent])[0]
                distances, _ = self.semantic_index.search(np.array([embedding]).astype('float32'), k=1)
                similarity = 1 / (1 + distances[0][0])
                if similarity > 0.85:
                    matches.append({'sentence': sent[:100], 'similarity': similarity})
        
        score = len(matches) / len(sentences)
        explanations = []
        if matches:
            explanations.append(f"Found {len(matches)} similar passages")
        if score > 0.3:
            explanations.append(f"Similarity score: {score*100:.1f}%")
        else:
            explanations.append("No significant matches found")
        
        return score, explanations, {'matches': matches}

# ============================================================================
# AUTHORSHIP VERIFIER
# ============================================================================

class AuthorshipVerifier:
    def __init__(self, db_path: str = "integrity.db"):
        self.db_path = db_path
        self.detector = AIDetector()
    
    def verify(self, student_id: str, text: str) -> Tuple[float, List[str], Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''SELECT features FROM submissions 
                         WHERE student_id = ? AND submission_type != 'final'
                         ORDER BY created_at DESC LIMIT 5''', (student_id,))
        previous = cursor.fetchall()
        conn.close()
        
        if not previous:
            return 0.0, ["No baseline submissions found"], {'baseline_count': 0}
        
        current = self.detector.extract_features(text)
        deviations = []
        
        for row in previous:
            try:
                prev = StylometricFeatures.from_json(row[0])
                dev = abs(current.lexical_diversity - prev.lexical_diversity) + \
                      abs(current.function_word_ratio - prev.function_word_ratio) + \
                      abs(current.hedging_density - prev.hedging_density)
                deviations.append(dev)
            except:
                continue
        
        if not deviations:
            return 0.0, ["Error comparing styles"], {'baseline_count': len(previous)}
        
        drift = min(np.mean(deviations) / 1.5, 1.0)
        explanations = []
        if drift > 0.6:
            explanations.append(f"Significant style deviation ({drift*100:.0f}% from baseline)")
        elif drift > 0.3:
            explanations.append(f"Moderate style deviation ({drift*100:.0f}% from baseline)")
        else:
            explanations.append("Writing style consistent with baseline")
        
        return drift, explanations, {'baseline_count': len(previous), 'deviation': drift}

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_risk_gauge(score: float):
    color = "darkred" if score > 70 else "orange" if score > 40 else "green"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_radar(metrics: Dict):
    categories = ['Lexical', 'Function', 'Hedging', 'Entropy', 'Burst', 'Perplexity']
    values = [metrics.get('Lexical Diversity', 0), metrics.get('Function Words', 0),
              metrics.get('Hedging Density', 0), metrics.get('Entropy', 0),
              metrics.get('Burstiness', 0), metrics.get('Perplexity', 0)]
    
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=400)
    return fig

def create_word_chart(text: str):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    word_counts = Counter(words).most_common(10)
    fig = go.Figure(data=go.Bar(x=[w[0] for w in word_counts], y=[w[1] for w in word_counts]))
    fig.update_layout(title="Top 10 Words", height=400)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

st.set_page_config(page_title="Integrity System", page_icon="🎓", layout="wide")

st.title("🎓 Academic Integrity Intelligence System")

# Initialize
@st.cache_resource
def init():
    return {
        'ai': AIDetector(),
        'plagiarism': PlagiarismEngine(),
        'authorship': AuthorshipVerifier()
    }

try:
    system = init()
except Exception as e:
    st.error(f"Init error: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    student_id = st.text_input("Student ID", placeholder="S12345")
    sub_type = st.selectbox("Submission Type", ["final", "draft1", "draft2"])
    uploaded = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt'])

# Main
if uploaded:
    # Extract
    with st.spinner("Processing..."):
        text = extract_text(uploaded.getvalue(), uploaded.type)
    
    if len(text.strip()) < 100:
        st.warning("Document too short")
        st.stop()
    
    # Show preview
    with st.expander("Text Preview"):
        st.text(text[:500] + ("..." if len(text) > 500 else ""))
    
    # Save button
    if student_id and st.sidebar.button("Save to Database"):
        system['plagiarism'].add_submission(student_id, sub_type, text)
        st.success("Saved!")
    
    # Analyze
    with st.spinner("Analyzing..."):
        ai_score, ai_explanations, ai_metrics = system['ai'].detect(text)
        plag_score, plag_explanations, plag_details = system['plagiarism'].check(text)
        
        if student_id:
            auth_drift, auth_explanations, auth_details = system['authorship'].verify(student_id, text)
        else:
            auth_drift, auth_explanations, auth_details = 0.0, ["No student ID"], {}
    
    # Calculate overall risk
    overall = (ai_score * 0.4 + plag_score * 0.35 + auth_drift * 0.25) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Risk", f"{overall:.0f}/100")
    with col2:
        st.metric("AI Likelihood", f"{ai_score*100:.0f}%")
    with col3:
        st.metric("Plagiarism", f"{plag_score*100:.0f}%")
    with col4:
        st.metric("Authorship", f"{auth_drift*100:.0f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_risk_gauge(overall), use_container_width=True)
    with col2:
        st.plotly_chart(create_radar(ai_metrics), use_container_width=True)
    
    # Word chart
    st.plotly_chart(create_word_chart(text), use_container_width=True)
    
    # Explanations
    st.subheader("Evidence")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Detection**")
        for e in ai_explanations:
            st.markdown(f"- {e}")
        st.markdown("**Plagiarism**")
        for e in plag_explanations:
            st.markdown(f"- {e}")
    
    with col2:
        st.markdown("**Authorship**")
        for e in auth_explanations:
            st.markdown(f"- {e}")
    
    # Risk level
    if overall > 70:
        st.error("🚨 HIGH RISK - Review required")
    elif overall > 40:
        st.warning("⚠️ MODERATE RISK - Further investigation suggested")
    else:
        st.success("✅ LOW RISK - Document appears authentic")
    
    # Export
    if st.button("Export Report"):
        report = f"""
INTEGRITY REPORT
===============
Student: {student_id or 'Anonymous'}
Date: {datetime.now()}

Scores:
- Overall Risk: {overall:.0f}/100
- AI Likelihood: {ai_score*100:.0f}%
- Plagiarism: {plag_score*100:.0f}%
- Authorship Drift: {auth_drift*100:.0f}%

Findings:
{chr(10).join(['- ' + e for e in ai_explanations])}
{chr(10).join(['- ' + e for e in plag_explanations])}
{chr(10).join(['- ' + e for e in auth_explanations])}

Recommendation: {"Review required" if overall > 70 else "Monitor" if overall > 40 else "Accept"}
"""
        st.download_button("Download", report, "report.txt")

else:
    st.info("👈 Upload a document to begin")
    
    st.markdown("""
    ### Features
    - **AI Detection** - Stylometric pattern analysis
    - **Plagiarism Check** - Semantic matching against database
    - **Authorship Verification** - Compare with previous submissions
    - **Interactive Visualizations** - Radar charts, gauges, word clouds
    - **Exportable Reports** - PDF/TXT downloads
    
    ### How to Use
    1. Enter Student ID (optional)
    2. Upload PDF, DOCX, or TXT
    3. Review analysis
    4. Save to database for future comparison
    """)