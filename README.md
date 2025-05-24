# NLP-Transformers----project
AI Resume Scanner & Job Matcher

resume-job-matcher/  
├── app.py                  # Streamlit web app  
├── requirements.txt        # Dependencies  
├── README.md               # Project documentation  
└── data/                   # Sample resumes & job descriptions  
    ├── resumes/  
    └── job_descriptions/


# 🤖 AI Resume Matcher  
*What it does*: Uses DistilBERT to extract skills from resumes & matches with job descriptions via cosine similarity.  
*How to run*: pip install -r requirements.txt then streamlit run app.py. Upload PDF/TXT resume.  
*Dataset*: Add resumes in data/resumes/ and job descriptions in data/job_descriptions/.  
*Live Demo*: [HuggingFace Spaces](https://huggingface.co/spaces).  
*Tech*: Python, Transformers, Streamlit, Scikit-learn.

This project leverages DistilBERT to extract relevant skills from resumes and match them with job descriptions using cosine similarity. Built with Streamlit, users can upload PDF/TXT resumes and instantly get job recommendations based on content similarity. It supports easy customization with your own dataset placed under the data/ folder.

Use Cases:

Job Portals: Automatically match candidate resumes with job listings.

Recruiters: Filter and shortlist resumes based on job descriptions.

Students/Freshers: Get instant feedback on job-fit for your resume.

HR Chatbots: Integrate the backend for smart resume screening.
