# AI-Powered Legal Document Analyzer

An AI-powered web application for analyzing legal documents using Natural Language Processing (NLP), Machine Learning, and Named Entity Recognition (NER).

## 📌 Project Overview

The AI-Powered Legal Document Analyzer is a Streamlit-based application that allows users to upload legal documents in PDF format and analyze their contents.

The system extracts text from legal documents, identifies important legal entities, detects relationships between statements, performs Natural Language Inference (NLI), and generates a risk assessment and analysis summary.

## 🚀 Key Features

- 📄 Upload legal documents in PDF format
- 🔍 Extract text from uploaded documents
- 🏷️ Legal Named Entity Recognition (NER)
- 🧠 Natural Language Inference (NLI)
- ⚖️ Contract statement analysis
- 🔎 Detection of contradictions and inconsistencies
- 📊 Risk score calculation
- 📋 Analysis summary
- 💡 Contract reasoning and interactive results

## 🛠️ Technologies Used

- Python
- Streamlit
- Natural Language Processing (NLP)
- Machine Learning
- spaCy
- Scikit-learn
- TF-IDF
- Logistic Regression
- PDF text extraction
- Joblib

## 🏗️ Project Structure

```text
AI-Powered-Legal-Document-Analyzer/
│
├── application/
│   └── app.py
│
├── training/
│   ├── saved_models/
│   ├── scripts/
│   └── download_dataset.py
│
├── README.md
├── requirements.txt
└── .gitignore
