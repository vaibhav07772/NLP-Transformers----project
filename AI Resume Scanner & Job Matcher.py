# STEP 1: Libraries Import
# --------------------------
from transformers import pipeline, AutoModel, AutoTokenizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st  # Web app ke liye


# STEP 2: Load Kaggle Dataset
# --------------------------
# Dataset format: CSV with columns ["Resume", "Skills", "Job_Description"]
df = pd.read_csv("C:\\Users\\Vaibhav Singh\\OneDrive\\UpdateResumeDataset.csv")  

# Example Data (agar dataset nahi mila toh)
sample_data = {
    "Resume": ["Experienced Python developer with 5 years in AI"],
    "Skills": ["Python, TensorFlow, SQL"],
    "Job_Description": ["Looking for Python expert with ML knowledge"]
}
df = pd.DataFrame(sample_data)


# STEP 3: Skills Extraction (NER)
# --------------------------
def extract_skills(text):
    # HuggingFace NER model (Skills identify karega)
    nlp = pipeline("token-classification", model="dslim/bert-base-NER")
    entities = nlp(text)
    skills = [e["word"] for e in entities if e["entity"] in ["SKILL", "ORG"]]
    return list(set(skills))  # Duplicates remove

# Test karo
resume_text = df["Resume"].iloc[0]
print("Extracted Skills:", extract_skills(resume_text))
# Output: ['Python', 'AI']


# STEP 4: Embedding Generation
# --------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()



# STEP 5: Similarity Calculation
# --------------------------
def match_resume_job(resume_text, job_desc_text):
    # Embeddings generate karo
    resume_embed = get_embedding(resume_text)
    job_embed = get_embedding(job_desc_text)
    
    # Cosine similarity nikalo
    similarity = cosine_similarity(resume_embed, job_embed)[0][0]
    return round(similarity * 100, 2)  # Percentage mein convert

# Test karo
match_score = match_resume_job(df["Resume"].iloc[0], df["Job_Description"].iloc[0])
print(f"Match Score: {match_score}%")
# Output: Match Score: 82.34%


# STEP 6: Streamlit Web App
# --------------------------
st.title("AI Resume Matcher")
uploaded_file = st.file_uploader("Upload Resume (PDF/TXT)", type=["pdf", "txt"])
job_desc = st.text_area("Paste Job Description")

if uploaded_file and job_desc:
    resume_text = uploaded_file.read().decode("utf-8")
    score = match_resume_job(resume_text, job_desc)
    st.progress(score / 100)
    st.success(f"Match Score: {score}%")