import streamlit as st
import fitz
import joblib
import spacy


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f5f5;
        margin-bottom: 15px;
    }

    .risk-high {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffe5e5;
        font-size: 20px;
        font-weight: 600;
    }

    .risk-medium {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff3cd;
        font-size: 20px;
        font-weight: 600;
    }

    .risk-low {
        padding: 15px;
        border-radius: 10px;
        background-color: #e5f7e5;
        font-size: 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">⚖️ AI-Powered Legal Document Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze legal contracts using Machine Learning, NLP and AI'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():
    nli_model = joblib.load(
        "training/saved_models/nli_classifier.pkl"
    )

    vectorizer = joblib.load(
        "training/saved_models/tfidf_vectorizer.pkl"
    )

    ner_model = spacy.load(
        "training/saved_models/legal_ner_model"
    )

    return nli_model, vectorizer, ner_model


try:
    nli_model, vectorizer, ner_model = load_models()
    models_loaded = True

except Exception as error:
    models_loaded = False
    st.error(
        f"Unable to load trained models: {error}"
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Analysis Options")

    st.write("This application uses:")
    st.write("🧠 TF-IDF + Logistic Regression")
    st.write("🔎 Legal Named Entity Recognition")
    st.write("📄 PDF text extraction")
    st.write("⚠️ Rule-based risk scoring")

    st.divider()

    st.info(
        "Upload a legal contract PDF to begin analysis."
    )


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload Legal Contract",
    type=["pdf"]
)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_file):
    document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    pages_text = []

    for page in document:
        pages_text.append(page.get_text())

    document.close()

    return "\n".join(pages_text)


# =========================================================
# NLI ANALYSIS
# =========================================================

def analyze_nli(text):
    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if len(sentence.strip()) > 20
    ]

    results = []

    hypotheses = [
        "The agreement contains confidential information.",
        "The agreement contains a payment obligation.",
        "The agreement contains a termination condition.",
        "The agreement contains information about the parties.",
        "The agreement contains a financial obligation."
    ]

    for sentence in sentences[:30]:
        for hypothesis in hypotheses:

            combined_text = (
                sentence
                + " [SEP] "
                + hypothesis
            )

            vectorized = vectorizer.transform(
                [combined_text]
            )

            prediction = nli_model.predict(
                vectorized
            )[0]

            results.append(
                {
                    "sentence": sentence,
                    "hypothesis": hypothesis,
                    "prediction": prediction
                }
            )

    return results


# =========================================================
# NER ANALYSIS
# =========================================================

def analyze_entities(text):
    doc = ner_model(text)

    entities = []

    for entity in doc.ents:
        entities.append(
            {
                "text": entity.text,
                "label": entity.label_
            }
        )

    return entities


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk(nli_results, entities):

    contradiction_count = sum(
        1
        for result in nli_results
        if result["prediction"].lower() == "contradiction"
    )

    financial_entities = sum(
        1
        for entity in entities
        if entity["label"].lower() in [
            "salary",
            "price",
            "rent",
            "percentage"
        ]
    )

    score = (
        contradiction_count * 15
        + financial_entities * 5
    )

    score = min(score, 100)

    if score >= 60:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


# =========================================================
# SESSION STATE
# =========================================================

if "nli_results" not in st.session_state:
    st.session_state.nli_results = []

if "entities" not in st.session_state:
    st.session_state.entities = []

if "text" not in st.session_state:
    st.session_state.text = ""

if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0

if "risk_level" not in st.session_state:
    st.session_state.risk_level = "LOW"

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# =========================================================
# ANALYZE BUTTON
# =========================================================

if uploaded_file is not None and models_loaded:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🔍 Analyze Contract",
        type="primary"
    ):

        with st.spinner(
            "AI is analyzing your contract..."
        ):

            # Extract text
            text = extract_text_from_pdf(
                uploaded_file
            )

            if not text.strip():
                st.error(
                    "No readable text was found in the PDF."
                )
                st.stop()

            # NER
            entities = analyze_entities(text)

            # NLI
            nli_results = analyze_nli(text)

            # Risk
            risk_score, risk_level = calculate_risk(
                nli_results,
                entities
            )

            # Save results in session state
            st.session_state.text = text
            st.session_state.entities = entities
            st.session_state.nli_results = nli_results
            st.session_state.risk_score = risk_score
            st.session_state.risk_level = risk_level
            st.session_state.analysis_done = True

        st.success(
            "Contract analysis completed!"
        )


# =========================================================
# DISPLAY RESULTS
# =========================================================

if st.session_state.analysis_done:

    text = st.session_state.text
    entities = st.session_state.entities
    nli_results = st.session_state.nli_results
    risk_score = st.session_state.risk_score
    risk_level = st.session_state.risk_level


    # =====================================================
    # SUMMARY
    # =====================================================

    st.header("📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pages / Document",
            "PDF"
        )

    with col2:
        st.metric(
            "Entities Detected",
            len(entities)
        )

    with col3:
        contradictions = sum(
            1
            for result in nli_results
            if result["prediction"].lower() == "contradiction"
        )

        st.metric(
            "Contradictions",
            contradictions
        )

    with col4:
        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    st.header("⚠️ Risk Assessment")

    if risk_level == "HIGH":

        st.markdown(
            '<div class="risk-high">'
            '🔴 HIGH RISK'
            '</div>',
            unsafe_allow_html=True
        )

    elif risk_level == "MEDIUM":

        st.markdown(
            '<div class="risk-medium">'
            '🟡 MEDIUM RISK'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="risk-low">'
            '🟢 LOW RISK'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # NER RESULTS
    # =====================================================

    st.header("🔎 Detected Legal Entities")

    if entities:

        for entity in entities:

            st.markdown(
                f"""
                <div class="result-box" style="color:#1f2937; background-color:#ffffff;">
                <b>{entity["label"]}</b>
                <br>
                {entity["text"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No legal entities were detected."
        )


    # =====================================================
    # NLI RESULTS
    # =====================================================

    st.header("🧠 Contract Reasoning")

    for result in nli_results[:15]:

        prediction = result["prediction"]

        if prediction.lower() == "contradiction":
            icon = "🔴"
            border_color = "#ef4444"

        elif prediction.lower() == "entailment":
            icon = "🟢"
            border_color = "#22c55e"

        else:
            icon = "🟡"
            border_color = "#eab308"

        st.markdown(
    f"""<div style="background-color:#1f2937;
color:#f8fafc;
padding:22px;
margin:18px 0;
border-radius:12px;
border-left:6px solid {border_color};">

<h4 style="color:#f8fafc; margin-top:0;">
{icon} {prediction}
</h4>

<div style="color:#e5e7eb; font-size:16px; margin-bottom:18px;">
<b>Contract statement:</b><br>
{result["sentence"]}
</div>

<div style="color:#e5e7eb; font-size:16px;">
<b>Hypothesis:</b><br>
{result["hypothesis"]}
</div>

</div>""",
    unsafe_allow_html=True
)


    # =====================================================
    # EXTRACTED TEXT
    # =====================================================

    with st.expander(
        "📄 View Extracted Contract Text"
    ):
        # Use st.code instead of st.text_area.
        # This avoids duplicate Streamlit widget-ID errors.
        st.code(
            text,
            language=None
        )

else:

    if not uploaded_file:

        st.info(
            "👆 Upload a PDF contract above to start."
        )