import streamlit as st
import hashlib
import sqlite3
import re
import io
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF extraction
try:
    import pypdf
    import pdfplumber
    from docx import Document
except ImportError as e:
    st.error(f"Please install required packages. Error: {e}")
    st.stop()

# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes"""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except:
        pass
    
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except:
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file bytes"""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text
    except:
        return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT file bytes"""
    try:
        return file_bytes.decode('utf-8')
    except:
        try:
            return file_bytes.decode('latin-1')
        except:
            return ""

def extract_text(file_bytes: bytes, file_type: str) -> str:
    """Main extraction dispatcher"""
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)

# ============================================================================
# AI DETECTION ENGINE (Pure Python, No ML)
# ============================================================================

class AIDetector:
    def __init__(self):
        # Function words (common grammatical words)
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
        }
        
        # Hedging words (cautious language)
        self.hedging_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'would', 'may',
            'suggests', 'indicates', 'appears', 'seems', 'likely', 'unclear',
            'uncertain', 'potential', 'tentative', 'speculative', 'often',
            'frequently', 'typically', 'generally', 'usually', 'approximately'
        }
        
        # AI-typical transition words
        self.ai_transitions = {
            'however', 'therefore', 'furthermore', 'consequently', 'additionally',
            'moreover', 'nevertheless', 'nonetheless', 'accordingly', 'subsequently',
            'hence', 'thus', 'conversely', 'alternatively', 'similarly', 'likewise'
        }
    
    def calculate_lexical_diversity(self, text: str) -> float:
        """Type-Token Ratio - unique words / total words"""
        words = text.lower().split()
        if len(words) < 10:
            return 0.5
        return len(set(words)) / len(words)
    
    def calculate_function_word_ratio(self, text: str) -> float:
        """Proportion of function words (higher in formal/AI text)"""
        words = text.lower().split()
        if not words:
            return 0.5
        func_count = sum(1 for w in words if w in self.function_words)
        return func_count / len(words)
    
    def calculate_hedging_density(self, text: str) -> float:
        """Proportion of hedging words (higher in human writing)"""
        words = text.lower().split()
        if not words:
            return 0.05
        hedge_count = sum(1 for w in words if w in self.hedging_words)
        return hedge_count / len(words)
    
    def calculate_ai_transition_density(self, text: str) -> float:
        """Proportion of AI-typical transitions (higher in AI text)"""
        words = text.lower().split()
        if len(words) < 20:
            return 0
        trans_count = sum(1 for w in words if w in self.ai_transitions)
        # Normalize to 0-1 range (typical AI has ~2-3% transitions)
        return min(trans_count / (len(words) * 0.03), 1.0)
    
    def calculate_sentence_length_variance(self, text: str) -> float:
        """Variance in sentence length (lower in AI text)"""
        sentences = re.split(r'[.!?]+', text)
        sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        
        if len(sent_lengths) < 3:
            return 0.5
        
        variance = np.var(sent_lengths)
        # Normalized to 0-1 (higher variance = more human-like)
        normalized = min(variance / 50, 1.0)
        return normalized
    
    def calculate_repetition_rate(self, text: str) -> float:
        """Frequency of repeated word sequences (higher in AI)"""
        words = text.lower().split()
        if len(words) < 30:
            return 0
        
        # Check 3-gram repetition
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(' '.join(words[i:i+3]))
        
        if not trigrams:
            return 0
        
        unique_ratio = len(set(trigrams)) / len(trigrams)
        # Lower unique ratio = more repetition = more AI-like
        return 1 - unique_ratio
    
    def detect(self, text: str) -> Tuple[float, List[str], Dict]:
        """Main detection method"""
        explanations = []
        
        # Calculate all features
        lexical_div = self.calculate_lexical_diversity(text)
        func_ratio = self.calculate_function_word_ratio(text)
        hedging = self.calculate_hedging_density(text)
        transitions = self.calculate_ai_transition_density(text)
        sent_variance = self.calculate_sentence_length_variance(text)
        repetition = self.calculate_repetition_rate(text)
        
        # Convert to AI scores (higher = more AI-like)
        ai_lexical = 1 - min(lexical_div / 0.6, 1.0)
        ai_func = min(func_ratio / 0.5, 1.0)
        ai_hedging = 1 - min(hedging / 0.08, 1.0)
        ai_transitions = transitions
        ai_variance = 1 - sent_variance
        ai_repetition = repetition
        
        # Weighted average
        weights = {
            'lexical': 0.20,
            'function': 0.15,
            'hedging': 0.20,
            'transitions': 0.20,
            'variance': 0.15,
            'repetition': 0.10
        }
        
        ai_score = (
            ai_lexical * weights['lexical'] +
            ai_func * weights['function'] +
            ai_hedging * weights['hedging'] +
            ai_transitions * weights['transitions'] +
            ai_variance * weights['variance'] +
            ai_repetition * weights['repetition']
        )
        
        # Build human-readable explanations
        if lexical_div < 0.4:
            explanations.append(f"⚠️ Low vocabulary diversity ({lexical_div:.2f}) - AI-like pattern")
        elif lexical_div > 0.6:
            explanations.append(f"✅ High vocabulary diversity ({lexical_div:.2f}) - human-like")
        
        if hedging < 0.02:
            explanations.append("⚠️ Very low hedging language - overly confident (AI pattern)")
        elif hedging > 0.05:
            explanations.append("✅ Natural hedging present - human-like caution")
        
        if transitions > 0.03:
            explanations.append(f"⚠️ High AI transition word density ({transitions:.3f})")
        
        if sent_variance < 0.3:
            explanations.append("⚠️ Uniform sentence length - AI generation pattern")
        
        if repetition > 0.4:
            explanations.append(f"⚠️ High phrase repetition ({repetition:.2f}) - AI pattern")
        
        if not explanations:
            explanations.append("📝 Text shows natural writing patterns")
        
        # Metrics for visualization
        metrics = {
            'AI Score': ai_score,
            'Lexical Diversity': lexical_div,
            'Function Words': func_ratio,
            'Hedging': hedging,
            'AI Transitions': transitions,
            'Sentence Variance': sent_variance,
            'Repetition': repetition
        }
        
        return ai_score, explanations, metrics

# ============================================================================
# SIMPLE PLAGIARISM DETECTION
# ============================================================================

class PlagiarismDetector:
    def __init__(self, db_path: str = "submissions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                filename TEXT,
                text_hash TEXT UNIQUE,
                text_sample TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Text chunks for matching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_id INTEGER,
                chunk_hash TEXT,
                chunk_text TEXT,
                FOREIGN KEY (submission_id) REFERENCES submissions (id)
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_hash ON text_chunks(chunk_hash)')
        
        conn.commit()
        conn.close()
    
    def add_submission(self, student_id: str, filename: str, text: str) -> bool:
        """Add a document to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Check if already exists
        cursor.execute("SELECT id FROM submissions WHERE text_hash = ?", (text_hash,))
        if cursor.fetchone():
            conn.close()
            return False
        
        # Insert submission
        cursor.execute(
            "INSERT INTO submissions (student_id, filename, text_hash, text_sample) VALUES (?, ?, ?, ?)",
            (student_id, filename, text_hash, text[:500])
        )
        sub_id = cursor.lastrowid
        
        # Insert text chunks (sentences)
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30:  # Only store meaningful chunks
                chunk_hash = hashlib.sha256(sent.encode()).hexdigest()
                cursor.execute(
                    "INSERT INTO text_chunks (submission_id, chunk_hash, chunk_text) VALUES (?, ?, ?)",
                    (sub_id, chunk_hash, sent[:200])
                )
        
        conn.commit()
        conn.close()
        return True
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def check(self, text: str, threshold: float = 0.7) -> Tuple[float, List[str], Dict]:
        """Check text against database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get sentences from current text
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        
        if not sentences:
            conn.close()
            return 0.0, ["No substantial sentences to analyze"], {'matches': []}
        
        matches = []
        
        for sent in sentences[:30]:  # Limit for performance
            sent_hash = hashlib.sha256(sent.encode()).hexdigest()
            
            # Check exact matches
            cursor.execute("SELECT submission_id, chunk_text FROM text_chunks WHERE chunk_hash = ?", (sent_hash,))
            exact = cursor.fetchone()
            
            if exact:
                matches.append({
                    'sentence': sent[:100],
                    'similarity': 1.0,
                    'type': 'exact'
                })
                continue
            
            # Check fuzzy matches (sample of database)
            cursor.execute("SELECT chunk_text FROM text_chunks ORDER BY RANDOM() LIMIT 100")
            for row in cursor.fetchall():
                similarity = self.calculate_similarity(sent, row[0])
                if similarity > threshold:
                    matches.append({
                        'sentence': sent[:100],
                        'similarity': similarity,
                        'type': 'similar'
                    })
                    break
        
        conn.close()
        
        # Calculate overall score
        score = len(matches) / len(sentences) if sentences else 0
        
        explanations = []
        if matches:
            explanations.append(f"📌 Found {len(matches)} matching or similar passages")
            explanations.append(f"Similarity score: {score*100:.1f}%")
        else:
            explanations.append("✅ No significant matches found in database")
        
        return score, explanations, {'matches': matches, 'total_sentences': len(sentences)}

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_risk_gauge(score: float):
    """Create a gauge chart for risk score"""
    color = "darkred" if score > 70 else "orange" if score > 40 else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Integrity Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_radar_chart(metrics: Dict):
    """Create radar chart for stylometric features"""
    categories = ['Lexical\nDiversity', 'Function\nWords', 'Hedging', 
                  'AI\nTransitions', 'Sentence\nVariance', 'Repetition']
    
    values = [
        metrics.get('Lexical Diversity', 0),
        metrics.get('Function Words', 0),
        metrics.get('Hedging', 0),
        metrics.get('AI Transitions', 0),
        metrics.get('Sentence Variance', 0),
        metrics.get('Repetition', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='rgba(255, 99, 71, 0.8)'),
        name='Current Document'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Stylometric Profile",
        height=400
    )
    
    return fig

def create_word_frequency_chart(text: str, top_n: int = 12):
    """Create bar chart of most frequent words"""
    # Clean and tokenize
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    if not words:
        return None
    
    # Count frequencies
    word_counts = Counter(words).most_common(top_n)
    
    fig = go.Figure(data=go.Bar(
        x=[w[0] for w in word_counts],
        y=[w[1] for w in word_counts],
        marker_color='teal',
        text=[w[1] for w in word_counts],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Frequent Words",
        xaxis_title="Word",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_sentence_length_chart(text: str):
    """Create histogram of sentence lengths"""
    sentences = re.split(r'[.!?]+', text)
    sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    
    if not sent_lengths:
        return None
    
    fig = go.Figure(data=go.Histogram(
        x=sent_lengths,
        nbinsx=15,
        marker_color='blue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Sentence Length Distribution",
        xaxis_title="Words per Sentence",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Academic Integrity System",
    page_icon="🎓",
    layout="wide"
)

# Initialize detectors
@st.cache_resource
def init_detectors():
    return {
        'ai': AIDetector(),
        'plagiarism': PlagiarismDetector()
    }

detectors = init_detectors()

# Custom CSS
st.markdown("""
<style>
.risk-high {
    background-color: #ffcccc;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid red;
}
.risk-medium {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid orange;
}
.risk-low {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid green;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Academic Integrity Intelligence System")
st.markdown("*AI Detection + Plagiarism Checking + Explainable Reports*")

# Sidebar
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Supports PDF, DOCX, and TXT files"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Database Status")
    
    # Show database count
    conn = sqlite3.connect("submissions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM submissions")
    doc_count = cursor.fetchone()[0]
    conn.close()
    
    st.info(f"📚 {doc_count} documents in database")
    
    # Save to database option
    if uploaded_file:
        student_id = st.text_input("Student ID (optional)", placeholder="e.g., S12345")
        if st.button("💾 Save to Database"):
            with st.spinner("Saving..."):
                text = extract_text(uploaded_file.getvalue(), uploaded_file.type)
                if text:
                    detectors['plagiarism'].add_submission(student_id or "anonymous", uploaded_file.name, text)
                    st.success("Document saved to database!")
                    st.rerun()

# Main content
if uploaded_file:
    # Extract text
    with st.spinner("Extracting text from document..."):
        text = extract_text(uploaded_file.getvalue(), uploaded_file.type)
    
    # Validate
    if not text or len(text.strip()) < 100:
        st.warning("⚠️ The document contains very little text. Please upload a longer document (at least 100 characters).")
        st.stop()
    
    # Show preview
    with st.expander("📄 Document Preview"):
        st.text(text[:1000] + ("..." if len(text) > 1000 else ""))
        st.caption(f"Total: {len(text):,} characters | {len(text.split()):,} words")
    
    # Run analysis
    with st.spinner("Analyzing document for integrity issues..."):
        ai_score, ai_explanations, ai_metrics = detectors['ai'].detect(text)
        plag_score, plag_explanations, plag_details = detectors['plagiarism'].check(text)
    
    # Calculate overall risk score
    overall_risk = (ai_score * 0.6 + plag_score * 0.4) * 100
    
    # Display metrics row
    st.markdown("## 📊 Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Risk", f"{overall_risk:.0f}/100", 
                  delta="High" if overall_risk > 70 else "Moderate" if overall_risk > 40 else "Low")
    
    with col2:
        st.metric("AI Likelihood", f"{ai_score*100:.0f}%")
    
    with col3:
        st.metric("Plagiarism Score", f"{plag_score*100:.0f}%")
    
    with col4:
        match_count = len(plag_details.get('matches', []))
        st.metric("Matches Found", match_count)
    
    # Visualizations row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_risk_gauge(overall_risk), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_radar_chart(ai_metrics), use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        word_chart = create_word_frequency_chart(text)
        if word_chart:
            st.plotly_chart(word_chart, use_container_width=True)
    
    with col2:
        sent_chart = create_sentence_length_chart(text)
        if sent_chart:
            st.plotly_chart(sent_chart, use_container_width=True)
    
    # Explanations
    st.markdown("## 🔍 Evidence & Explanations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 AI Detection Signals")
        for exp in ai_explanations:
            st.markdown(f"- {exp}")
    
    with col2:
        st.markdown("### 📋 Plagiarism Detection")
        for exp in plag_explanations:
            st.markdown(f"- {exp}")
        
        if plag_details.get('matches'):
            st.markdown(f"**Found {len(plag_details['matches'])} potential matches in database**")
            for i, match in enumerate(plag_details['matches'][:3]):
                st.markdown(f"- Match {i+1}: {match['sentence']}...")
    
    # Risk verdict
    st.markdown("---")
    
    if overall_risk > 70:
        st.markdown('<div class="risk-high">', unsafe_allow_html=True)
        st.error("🚨 HIGH RISK - Significant integrity concerns detected")
        st.markdown("**Recommendation:** Faculty review required. Multiple indicators suggest potential academic integrity issues.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif overall_risk > 40:
        st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
        st.warning("⚠️ MODERATE RISK - Several indicators require attention")
        st.markdown("**Recommendation:** Further investigation recommended. Consider student interview or additional review.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">', unsafe_allow_html=True)
        st.success("✅ LOW RISK - No major integrity concerns")
        st.markdown("**Recommendation:** Document appears authentic. No further action required.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Export report
    st.markdown("## 📥 Export")
    
    if st.button("Generate Report"):
        report = f"""
ACADEMIC INTEGRITY ASSESSMENT REPORT
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

OVERALL ASSESSMENT
------------------
Risk Score: {overall_risk:.0f}/100
Risk Level: {'HIGH' if overall_risk > 70 else 'MODERATE' if overall_risk > 40 else 'LOW'}
AI Likelihood: {ai_score*100:.1f}%
Plagiarism Score: {plag_score*100:.1f}%

AI DETECTION DETAILS
--------------------
Lexical Diversity: {ai_metrics['Lexical Diversity']:.3f}
Function Word Ratio: {ai_metrics['Function Words']:.3f}
Hedging Density: {ai_metrics['Hedging']:.3f}
AI Transition Density: {ai_metrics['AI Transitions']:.3f}
Sentence Variance: {ai_metrics['Sentence Variance']:.3f}
Repetition Rate: {ai_metrics['Repetition']:.3f}

FINDINGS
--------
AI Detection Signals:
{chr(10).join(['- ' + e for e in ai_explanations])}

Plagiarism Findings:
{chr(10).join(['- ' + e for e in plag_explanations])}

RECOMMENDATION
--------------
{"Immediate faculty review required" if overall_risk > 70 else "Further investigation recommended" if overall_risk > 40 else "Document accepted"}

DISCLAIMER
----------
This analysis is probabilistic and should be used as an investigative tool.
Final determination requires human review.
"""
        st.download_button(
            label="Download Report (TXT)",
            data=report,
            file_name=f"integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

else:
    # Welcome screen
    st.info("👈 Upload a document to begin analysis")
    
    st.markdown("""
    ### 🎓 Academic Integrity Intelligence System
    
    This system provides explainable, multi-dimensional analysis:
    
    #### 🤖 AI Detection
    - **Lexical Diversity** - Measures vocabulary richness
    - **Function Word Analysis** - Pattern detection
    - **Hedging Language** - Cautious vs. confident writing
    - **Sentence Structure** - Variance and complexity
    - **Repetition Patterns** - Phrase frequency analysis
    
    #### 📋 Plagiarism Detection
    - **Exact Matching** - Identical text detection
    - **Similarity Scoring** - Fuzzy matching algorithm
    - **Local Database** - Compare against stored submissions
    
    #### ⚖️ Risk Assessment
    - **Weighted Scoring** - Combines multiple signals
    - **Explainable Output** - Clear reasons for each flag
    - **Actionable Recommendations**
    
    ### How to Use
    1. Upload a PDF, DOCX, or TXT file
    2. Review the visual analysis
    3. Read the evidence explanations
    4. Export a report if needed
    5. Optionally save to database for future comparison
    
    ### Privacy Guarantee
    - All processing happens locally
    - No external API calls
    - Your documents stay private
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
🎓 Academic Integrity Intelligence System | Explainable | Privacy-First | Faculty-Ready
<br>
<small>This tool provides probabilistic analysis. Human review is required for any academic decisions.</small>
</div>
""", unsafe_allow_html=True)
