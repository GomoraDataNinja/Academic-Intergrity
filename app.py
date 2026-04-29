import streamlit as st
import hashlib
import sqlite3
import re
import math
import io
import pickle
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import pypdf
import pdfplumber
from docx import Document
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import faiss

# Optional ML imports (with fallback)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes"""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except:
        # Fallback to pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file bytes"""
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT file bytes"""
    return file_bytes.decode('utf-8')

def extract_text(file_bytes: bytes, file_type: str) -> str:
    """Main extraction dispatcher"""
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
    """Comprehensive stylometric features for authorship analysis"""
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
        """Convert to JSON string for storage"""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StylometricFeatures':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

@dataclass
class IntegrityRisk:
    """Multi-dimensional risk assessment"""
    overall_score: float
    ai_likelihood: float
    plagiarism_score: float
    authorship_drift: float
    citation_risk: float
    metadata_suspicion: float
    confidence: str
    explainability: List[str]

# ============================================================================
# ENHANCED AI DETECTION ENGINE
# ============================================================================

class EnhancedAIDetector:
    """Multi-signal AI detection with explainability"""
    
    def __init__(self):
        # Function words (high-frequency grammatical words)
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she'
        }
        
        # Hedging words (tentative language)
        self.hedging_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'would', 'may',
            'suggests', 'indicates', 'appears', 'seems', 'likely', 'unclear',
            'uncertain', 'potential', 'tentative', 'speculative'
        }
    
    def calculate_lexical_diversity(self, text: str) -> float:
        """Type-token ratio (unique words / total words)"""
        words = text.lower().split()
        if not words:
            return 0.0
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def calculate_function_word_ratio(self, text: str) -> float:
        """Proportion of function words in text"""
        words = text.lower().split()
        if not words:
            return 0.0
        function_count = sum(1 for w in words if w in self.function_words)
        return function_count / len(words)
    
    def calculate_hedging_density(self, text: str) -> float:
        """Frequency of hedging/cautious language"""
        words = text.lower().split()
        if not words:
            return 0.0
        hedging_count = sum(1 for w in words if w in self.hedging_words)
        return hedging_count / len(words)
    
    def calculate_punctuation_diversity(self, text: str) -> float:
        """Measure punctuation variety"""
        punct_marks = ['.', ',', ';', ':', '!', '?', '(', ')', '"', "'"]
        punct_counts = {p: text.count(p) for p in punct_marks}
        total_punct = sum(punct_counts.values())
        if total_punct == 0:
            return 0.0
        probs = [c/total_punct for c in punct_counts.values() if c > 0]
        entropy = -sum(p * np.log(p) for p in probs)
        return entropy / np.log(len(punct_marks)) if len(punct_marks) > 0 else 0
    
    def calculate_entropy(self, text: str) -> float:
        """Character-level entropy"""
        if not text:
            return 0.0
        chars = list(text.lower())
        char_counts = Counter(chars)
        probs = [count/len(chars) for count in char_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        max_entropy = np.log(len(char_counts)) if len(char_counts) > 0 else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def calculate_perplexity_score(self, text: str) -> float:
        """Estimate perplexity based on unigram diversity"""
        words = text.lower().split()
        if len(words) < 5:
            return 0.5
        word_counts = Counter(words)
        unique_ratio = len(word_counts) / len(words)
        perplexity_score = 1 - min(unique_ratio, 0.8) / 0.8
        return perplexity_score
    
    def calculate_burstiness(self, text: str) -> float:
        """Measure variance in sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        if len(sent_lengths) < 2:
            return 0.5
        cv = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0
        return 1 - min(cv, 0.5) / 0.5
    
    def extract_stylometric_features(self, text: str) -> StylometricFeatures:
        """Extract comprehensive stylometric features"""
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
            punctuation_diversity=self.calculate_punctuation_diversity(text),
            repetition_rate=1 - lexical_div,
            entropy=self.calculate_entropy(text)
        )
    
    def detect_with_explainability(self, text: str) -> Tuple[float, List[str]]:
        """Return AI score + list of reasons"""
        features = self.extract_stylometric_features(text)
        explanations = []
        
        signals = {
            'lexical_diversity': features.lexical_diversity,
            'function_word_ratio': features.function_word_ratio,
            'hedging_density': features.hedging_density,
            'entropy': features.entropy,
            'burstiness': self.calculate_burstiness(text),
            'perplexity': self.calculate_perplexity_score(text)
        }
        
        weights = {
            'lexical_diversity': 0.15,
            'function_word_ratio': 0.20,
            'hedging_density': 0.20,
            'entropy': 0.15,
            'burstiness': 0.15,
            'perplexity': 0.15
        }
        
        normalized_signals = {
            'lexical_diversity': 1 - signals['lexical_diversity'],
            'function_word_ratio': signals['function_word_ratio'],
            'hedging_density': 1 - signals['hedging_density'],
            'entropy': 1 - signals['entropy'],
            'burstiness': signals['burstiness'],
            'perplexity': signals['perplexity']
        }
        
        ai_score = sum(normalized_signals[k] * weights[k] for k in weights)
        
        # Build explanations
        if normalized_signals['lexical_diversity'] > 0.7:
            explanations.append("Unusually low lexical diversity (repetitive vocabulary)")
        elif normalized_signals['lexical_diversity'] > 0.5:
            explanations.append("Moderate lexical diversity")
            
        if normalized_signals['function_word_ratio'] > 0.6:
            explanations.append("High function word density (formal/AI-like pattern)")
            
        if normalized_signals['hedging_density'] < 0.3:
            explanations.append("Low hedging language (overly confident, AI-like)")
        elif normalized_signals['hedging_density'] > 0.6:
            explanations.append("High hedging density (cautious/human-like)")
            
        if normalized_signals['entropy'] < 0.4:
            explanations.append("Low character entropy (highly predictable text)")
            
        if normalized_signals['burstiness'] < 0.3:
            explanations.append("Uniform sentence length (AI generation pattern)")
        elif normalized_signals['burstiness'] > 0.7:
            explanations.append("High sentence length variance (human-like)")
            
        if normalized_signals['perplexity'] < 0.3:
            explanations.append("Low perplexity (predictable word choices)")
        
        return ai_score, explanations

# ============================================================================
# ENHANCED PLAGIARISM ENGINE
# ============================================================================

class EnhancedPlagiarismEngine:
    """Multi-method plagiarism detection with explainability"""
    
    def __init__(self, db_path: str = "academic_integrity.db"):
        self.db_path = db_path
        self.init_database()
        
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        # FAISS indexes
        self.semantic_index = None
        self.chunk_texts = []
        
        self.load_indexes()
    
    def init_database(self):
        """Initialize comprehensive database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Student profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE,
                baseline_features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                submission_type TEXT,
                text_hash TEXT UNIQUE,
                stylometric_features TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Plagiarism matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plagiarism_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_id INTEGER,
                match_type TEXT,
                source_doc TEXT,
                similarity_score REAL,
                matched_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Citation database
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citation_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citation_key TEXT UNIQUE,
                source TEXT,
                verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_indexes(self):
        """Load FAISS indexes"""
        try:
            self.semantic_index = faiss.read_index("semantic_index.faiss")
            with open("chunk_texts.pkl", "rb") as f:
                self.chunk_texts = pickle.load(f)
        except:
            self.semantic_index = faiss.IndexFlatL2(self.dimension)
            self.chunk_texts = []
    
    def save_indexes(self):
        """Save FAISS indexes"""
        faiss.write_index(self.semantic_index, "semantic_index.faiss")
        with open("chunk_texts.pkl", "wb") as f:
            pickle.dump(self.chunk_texts, f)
    
    def add_submission(self, student_id: str, submission_type: str, text: str):
        """Add student submission"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Extract stylometric features
        detector = EnhancedAIDetector()
        features = detector.extract_stylometric_features(text)
        
        # Store as JSON
        features_json = features.to_json()
        
        cursor.execute('''
            INSERT INTO submissions (student_id, submission_type, text_hash, stylometric_features)
            VALUES (?, ?, ?, ?)
        ''', (student_id, submission_type, text_hash, features_json))
        
        submission_id = cursor.lastrowid
        
        # Add to search index
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30:
                embedding = self.model.encode([sent])[0]
                self.chunk_texts.append(sent)
                self.semantic_index.add(np.array([embedding]).astype('float32'))
        
        conn.commit()
        conn.close()
        self.save_indexes()
        
        return submission_id
    
    def check_plagiarism_with_explainability(self, text: str) -> Tuple[float, List[str]]:
        """Full plagiarism check with explanations"""
        explanations = []
        matches = {
            'exact': [],
            'fuzzy': [],
            'semantic': []
        }
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        
        if not sentences:
            return 0.0, ["No substantial sentences to analyze"]
        
        matched_count = 0
        
        for sent in sentences[:50]:
            # Semantic match
            if self.semantic_index and self.semantic_index.ntotal > 0:
                embedding = self.model.encode([sent])[0]
                distances, indices = self.semantic_index.search(
                    np.array([embedding]).astype('float32'), k=1
                )
                similarity = 1 / (1 + distances[0][0])
                
                if similarity > 0.85:
                    matches['semantic'].append(sent)
                    matched_count += 1
                    continue
        
        plagiarism_score = matched_count / len(sentences) if sentences else 0
        
        # Build explanations
        if matches['exact']:
            explanations.append(f"Found {len(matches['exact'])} exact textual matches")
        
        if matches['semantic']:
            explanations.append(f"Found {len(matches['semantic'])} paraphrased matches")
        
        if plagiarism_score > 0.3:
            explanations.append(f"Overall similarity score: {plagiarism_score*100:.1f}%")
        
        if plagiarism_score < 0.1:
            explanations.append("No significant matches detected")
        
        return plagiarism_score, explanations

# ============================================================================
# AUTHORSHIP VERIFICATION
# ============================================================================

class AuthorshipVerifier:
    """Compare against student's previous work"""
    
    def __init__(self, db_path: str = "academic_integrity.db"):
        self.db_path = db_path
        self.detector = EnhancedAIDetector()
    
    def verify_authorship(self, student_id: str, current_text: str) -> Tuple[float, List[str]]:
        """Check if writing style matches student's baseline"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous submissions
        cursor.execute('''
            SELECT stylometric_features FROM submissions 
            WHERE student_id = ? AND submission_type != 'final'
            ORDER BY submitted_at DESC LIMIT 5
        ''', (student_id,))
        
        previous = cursor.fetchall()
        conn.close()
        
        if not previous:
            return 0.0, ["No baseline submissions available for comparison"]
        
        # Extract current features
        current_features = self.detector.extract_stylometric_features(current_text)
        
        # Compare with previous using JSON parsing
        deviations = []
        for prev_row in previous:
            try:
                # Parse JSON stored in database
                prev_features_dict = json.loads(prev_row[0])
                prev_features = StylometricFeatures(**prev_features_dict)
                
                feature_deviation = {
                    'lexical_diversity': abs(current_features.lexical_diversity - prev_features.lexical_diversity),
                    'function_word_ratio': abs(current_features.function_word_ratio - prev_features.function_word_ratio),
                    'hedging_density': abs(current_features.hedging_density - prev_features.hedging_density),
                    'entropy': abs(current_features.entropy - prev_features.entropy)
                }
                
                deviations.append(feature_deviation)
            except Exception as e:
                print(f"Error parsing features: {e}")
                continue
        
        if not deviations:
            return 0.0, ["Error comparing writing styles"]
        
        avg_deviation = np.mean([sum(d.values()) for d in deviations]) if deviations else 0
        authorship_drift = min(avg_deviation / 0.5, 1.0)
        
        # Build explanations
        explanations = []
        if authorship_drift > 0.6:
            explanations.append(f"Significant writing style deviation ({authorship_drift*100:.0f}% from baseline)")
            explanations.append("Stylometric markers differ substantially from previous submissions")
        elif authorship_drift > 0.3:
            explanations.append(f"Moderate style deviation ({authorship_drift*100:.0f}% from baseline)")
        else:
            explanations.append(f"Writing style consistent with baseline ({(1-authorship_drift)*100:.0f}% similar)")
        
        if current_features.lexical_diversity < 0.4:
            explanations.append("Lower vocabulary diversity than previous work")
        
        return authorship_drift, explanations

# ============================================================================
# RISK FUSION ENGINE
# ============================================================================

class RiskFusionEngine:
    """Combine all signals into final risk score"""
    
    def __init__(self):
        self.weights = {
            'ai_stylometry': 0.30,
            'plagiarism': 0.25,
            'authorship_drift': 0.25,
            'citation_risk': 0.10,
            'metadata_suspicion': 0.10
        }
    
    def calculate_risk(self, 
                      ai_score: float,
                      plagiarism_score: float,
                      authorship_drift: float,
                      citation_risk: float = 0.0,
                      metadata_suspicion: float = 0.0) -> IntegrityRisk:
        """Calculate weighted risk score"""
        
        overall = (
            self.weights['ai_stylometry'] * ai_score +
            self.weights['plagiarism'] * plagiarism_score +
            self.weights['authorship_drift'] * authorship_drift +
            self.weights['citation_risk'] * citation_risk +
            self.weights['metadata_suspicion'] * metadata_suspicion
        ) * 100
        
        # Determine confidence level
        if abs(ai_score - 0.5) > 0.3 or plagiarism_score > 0.4 or authorship_drift > 0.5:
            confidence = "High"
        elif abs(ai_score - 0.5) > 0.15 or plagiarism_score > 0.2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Build explanations
        explanations = []
        if ai_score > 0.6:
            explanations.append(f"AI likelihood: {ai_score*100:.0f}% (stylometric analysis)")
        elif ai_score < 0.3:
            explanations.append("Low AI indicators")
        
        if plagiarism_score > 0.3:
            explanations.append(f"Plagiarism score: {plagiarism_score*100:.1f}%")
        
        if authorship_drift > 0.5:
            explanations.append(f"Authorship drift: {authorship_drift*100:.0f}% from baseline")
        
        return IntegrityRisk(
            overall_score=round(overall, 1),
            ai_likelihood=round(ai_score, 2),
            plagiarism_score=round(plagiarism_score, 2),
            authorship_drift=round(authorship_drift, 2),
            citation_risk=citation_risk,
            metadata_suspicion=metadata_suspicion,
            confidence=confidence,
            explainability=explanations
        )

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Academic Integrity Intelligence System",
    page_icon="🎓",
    layout="wide"
)

# Initialize all components
@st.cache_resource
def init_system():
    return {
        'ai_detector': EnhancedAIDetector(),
        'plagiarism_engine': EnhancedPlagiarismEngine(),
        'authorship_verifier': AuthorshipVerifier(),
        'risk_fusion': RiskFusionEngine()
    }

try:
    system = init_system()
except Exception as e:
    st.error(f"Error initializing system: {str(e)}")
    st.info("Please make sure all dependencies are installed: pip install sentence-transformers faiss-cpu")
    st.stop()

# Custom CSS
st.markdown("""
<style>
.risk-high {
    background-color: #ffcccc;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid red;
}
.risk-medium {
    background-color: #fff3cd;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid orange;
}
.risk-low {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid green;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Academic Integrity Intelligence System")
st.markdown("*Multi-dimensional integrity assessment with explainable evidence*")

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Configuration")
    
    student_id = st.text_input("Student ID (for authorship verification)", placeholder="e.g., S12345")
    
    submission_type = st.selectbox(
        "Submission Type",
        ["final", "draft1", "draft2", "outline"],
        help="For progressive writing analysis"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Current Database")
    
    try:
        conn = sqlite3.connect("academic_integrity.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM submissions")
        sub_count = cursor.fetchone()[0]
        st.info(f"📝 {sub_count} submissions in database")
        conn.close()
    except:
        st.info("Database ready")
    
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'docx', 'txt'],
        help="PDF, DOCX, or TXT files"
    )

# Main content
if uploaded_file:
    # Extract text
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file.getvalue(), uploaded_file.type)
    
    if not text or len(text.strip()) < 50:
        st.warning("The document contains very little text. Please upload a document with more content.")
        st.stop()
    
    # Store in database if student ID provided
    if student_id and st.sidebar.button("Save to Database"):
        with st.spinner("Saving submission..."):
            system['plagiarism_engine'].add_submission(student_id, submission_type, text)
            st.success(f"Submission saved as {submission_type}")
            st.rerun()
    
    # Run all detections
    with st.spinner("Analyzing for integrity risks..."):
        # 1. AI Detection
        ai_score, ai_explanations = system['ai_detector'].detect_with_explainability(text)
        
        # 2. Plagiarism Detection
        plagiarism_score, plag_explanations = system['plagiarism_engine'].check_plagiarism_with_explainability(text)
        
        # 3. Authorship Verification
        if student_id:
            authorship_drift, authorship_explanations = system['authorship_verifier'].verify_authorship(student_id, text)
        else:
            authorship_drift = 0.0
            authorship_explanations = ["No student ID provided for authorship verification"]
        
        # 4. Risk Fusion
        risk = system['risk_fusion'].calculate_risk(
            ai_score=ai_score,
            plagiarism_score=plagiarism_score,
            authorship_drift=authorship_drift
        )
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Integrity Risk", f"{risk.overall_score}/100")
        st.caption(f"Confidence: {risk.confidence}")
    
    with col2:
        st.metric("AI Likelihood", f"{risk.ai_likelihood*100:.0f}%")
    
    with col3:
        st.metric("Plagiarism Score", f"{risk.plagiarism_score*100:.0f}%")
    
    # Risk level display
    st.markdown("---")
    if risk.overall_score > 70:
        st.markdown('<div class="risk-high">', unsafe_allow_html=True)
        st.error("🚨 HIGH RISK - Significant integrity concerns detected")
        st.markdown('</div>', unsafe_allow_html=True)
    elif risk.overall_score > 40:
        st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
        st.warning("⚠️ MODERATE RISK - Several indicators require review")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">', unsafe_allow_html=True)
        st.success("✅ LOW RISK - No major integrity concerns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Explainability section
    st.subheader("🔍 Evidence & Explanations")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("**🤖 AI Detection Signals**")
        for exp in ai_explanations:
            st.markdown(f"- {exp}")
        
        st.markdown("**📋 Plagiarism Detection**")
        for exp in plag_explanations:
            st.markdown(f"- {exp}")
    
    with col_exp2:
        st.markdown("**✍️ Authorship Verification**")
        for exp in authorship_explanations:
            st.markdown(f"- {exp}")
        
        if risk.explainability:
            st.markdown("**⚖️ Overall Risk Factors**")
            for exp in risk.explainability:
                st.markdown(f"- {exp}")
    
    # Download report
    if st.button("📥 Generate Investigation Report"):
        report = f"""
ACADEMIC INTEGRITY INVESTIGATION REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Student ID: {student_id or 'Not provided'}
Submission Type: {submission_type}

OVERALL RISK ASSESSMENT
-----------------------
Risk Score: {risk.overall_score}/100
Confidence Level: {risk.confidence}
Risk Category: {'HIGH' if risk.overall_score > 70 else 'MODERATE' if risk.overall_score > 40 else 'LOW'}

DIMENSIONS ANALYZED
-------------------
AI Likelihood: {risk.ai_likelihood*100:.1f}%
Plagiarism Score: {risk.plagiarism_score*100:.1f}%
Authorship Drift: {risk.authorship_drift*100:.1f}%

EVIDENCE SUMMARY
----------------
AI Detection Findings:
{chr(10).join(['- ' + e for e in ai_explanations])}

Plagiarism Findings:
{chr(10).join(['- ' + e for e in plag_explanations])}

Authorship Findings:
{chr(10).join(['- ' + e for e in authorship_explanations])}

RECOMMENDATION
--------------
{"This paper requires immediate review by academic integrity committee." if risk.overall_score > 70 else 
 "Further investigation recommended with student interview." if risk.overall_score > 40 else
 "No further action required. Paper shows normal academic integrity patterns."}

DISCLAIMER
----------
This report is probabilistic and should be used as an investigative tool, 
not as definitive proof of misconduct. Human review is required before any academic action.
"""
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"integrity_report_{student_id or 'anonymous'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
else:
    # Welcome
    st.info("👈 Upload a document and provide Student ID for full analysis")
    
    st.markdown("""
    ### 🎓 Academic Integrity Intelligence System
    
    This system provides **multi-dimensional integrity assessment** with explainable evidence.
    
    #### Core Capabilities
    
    1. **🤖 AI Detection Engine** - Stylometric feature analysis, perplexity, burstiness
    2. **📋 Plagiarism Detection** - Exact and semantic matching, paraphrase detection
    3. **✍️ Authorship Verification** - Compare against student's previous work
    4. **⚖️ Risk Fusion Engine** - Weighted multi-factor scoring with confidence
    
    #### Getting Started
    
    1. Enter a Student ID (optional but recommended)
    2. Upload a document (PDF, DOCX, or TXT)
    3. Review the multi-dimensional analysis
    4. Download the investigation report
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
🎓 Academic Integrity Intelligence System | Explainable | Probabilistic | Privacy-First
<br>
<small>Not a definitive detector — an investigative assistant for human reviewers</small>
</div>
""", unsafe_allow_html=True)