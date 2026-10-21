import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


load_dotenv()


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

def load_document(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# --------------------------------------------------
# FAISS VECTOR STORE
# --------------------------------------------------

def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# --------------------------------------------------
# SEARCH DOCUMENTS
# --------------------------------------------------

def search_documents(vector_store, query):
    results = vector_store.similarity_search(query, k=4)
    return results


# --------------------------------------------------
# LLM
# --------------------------------------------------

def create_llm():
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0
    )
    return llm


# --------------------------------------------------
# NORMAL ASK DOCUMENTS
# --------------------------------------------------

def generate_answer(llm, query, documents):
    context_parts = []

    for document in documents:
        source = os.path.basename(
            document.metadata.get("source", "Unknown document")
        )
        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
DOCUMENT: {source}
PAGE: {page}

CONTENT:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are DocLens AI.

Answer the user's question using ONLY the information
contained in the provided documents.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer cannot be found in the documents, say:
  "I couldn't find this information in the uploaded documents."
- Give a clear and concise answer.
- Mention the document and page when relevant.

DOCUMENT CONTEXT:

{context}

QUESTION:

{query}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response.content


# --------------------------------------------------
# COMPARE DOCUMENTS
# --------------------------------------------------

def generate_comparison(llm, documents):
    context_parts = []

    for document in documents:
        source = os.path.basename(
            document.metadata.get("source", "Unknown document")
        )
        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
DOCUMENT: {source}
PAGE: {page}

CONTENT:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are DocLens AI, an intelligent Resume vs Job
Description analysis system.

The uploaded documents contain a Job Description (JD)
and a Candidate Resume.

Your task is to deeply compare the candidate's resume
against the job description.

Use ONLY information found in the provided documents.

Do not invent:
- Skills
- Experience
- Projects
- Education
- Thesis
- Certifications
- Technologies
- Achievements

If information is not available, say:
"Not mentioned in the provided documents."

==================================================
1. OVERALL RESUME MATCH
==================================================

Calculate an estimated Resume-to-JD match percentage.

Use this scoring framework:

Technical Skills        = 35%
Experience              = 20%
Projects                = 20%
Education / Thesis      = 10%
Tools / Technologies    = 10%
Other Requirements      = 5%

Total = 100%

Show the category scores first.

Example:

Technical Skills: 30/35
Experience: 18/20
Projects: 16/20
Education / Thesis: 8/10
Tools / Technologies: 8/10
Other Requirements: 4/5

Overall Match: 84%

The score must be based on evidence from the documents.
Do not randomly choose a percentage.

==================================================
2. CANDIDATE STRENGTHS
==================================================

Identify the strongest parts of the candidate's profile.

For every important strength:
- Identify the JD requirement.
- Identify the resume evidence.
- Explain why the candidate's skill, experience,
  project, or education is relevant.

Be descriptive. Do not simply list technologies.

==================================================
3. SKILL COMPARISON
==================================================

### Strong Matches
List skills that are clearly required by the JD and clearly
demonstrated in the resume.

### Partial Matches
List requirements where the resume shows related knowledge
or experience but does not completely demonstrate the requirement.

### Missing / Not Clearly Mentioned
List important JD requirements that are not clearly found in the resume.

==================================================
4. RELEVANT PROJECTS
==================================================

For every relevant project explain:

Project:
Technologies:
What the candidate did:
Related JD requirement:
Why this project is relevant:

==================================================
5. EDUCATION AND THESIS
==================================================

Analyze degree, specialization, academic projects, thesis,
final-year project, and relevant coursework.

Explain how each relevant item connects to the JD.
If thesis information is not available, say:
"Thesis information is not mentioned in the provided resume."

==================================================
6. EXPERIENCE COMPARISON
==================================================

Compare the JD's experience requirement with the candidate's
documented experience.

Explain:
JD requirement:
Resume evidence:
Alignment:

Do not invent experience.

==================================================
7. WHY THIS PROFILE ALIGNS WITH THE ROLE
==================================================

Write a detailed explanation connecting technical skills, projects,
experience, education, thesis or academic work, and tools.

Do not simply say "The candidate is suitable." Explain the evidence.

==================================================
8. AREAS TO IMPROVE
==================================================

For every important gap provide:

Requirement from JD:
Evidence from JD:
Resume evidence:
Gap:
Suggested improvement:

==================================================
9. SUPPORTING EVIDENCE
==================================================

Show evidence separately.

### Job Description Evidence
Mention the document name and page number.

### Resume Evidence
Mention the document name and page number.

==================================================
10. FINAL SUMMARY
==================================================

Overall Match: XX%
Strongest Strength:
Most Relevant Project:
Strongest Skill Match:
Main Gap:
Most Important Improvement:

IMPORTANT:
- Base everything on the uploaded documents.
- Do not invent information or use outside knowledge.
- Do not randomly assign the match percentage.
- Treat the percentage as a document-based match score,
  not a prediction of hiring.
- Clearly distinguish the JD from the Resume.

UPLOADED DOCUMENTS:

{context}

Now provide the complete analysis.
"""

    response = llm.invoke(prompt)
    return response.content
