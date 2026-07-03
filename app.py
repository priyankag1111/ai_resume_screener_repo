
import streamlit as st
import os
import tempfile
from typing import Dict, List
from dotenv import load_dotenv # Import load_dotenv

from pydantic import BaseModel, Field, conint

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq

# --- 1. Structured output schemas (Copied from previous cells) ---
class ResumeEvaluation(
    BaseModel
):
    """Structured evaluation of a single resume against a job description."""
    candidate_name: str = Field(description="Candidate's name as found in the resume text; use 'Unknown' if it cannot be found")
    match_score: conint(ge=0, le=100) = Field(description="Overall match score of the resume against the JD, from 0 (no match) to 100 (perfect match)")
    matching_skills: List[str] = Field(description="Skills, tools, or qualifications required by the JD that ARE present in the resume")
    missing_skills: List[str] = Field(description="Skills, tools, or qualifications required by the JD that are NOT present in the resume")
    summary: str = Field(description="A concise 2-4 sentence summary of the candidate's background and experience")
    strengths: List[str] = Field(description="The candidate's key strengths relative to this specific job description")
    weaknesses: List[str] = Field(description="The candidate's key weaknesses or gaps relative to this specific job description")
    recommendation: str = Field(description="One of exactly: 'Strongly Recommend', 'Recommend', 'Consider', 'Not Recommended'")
    justification: str = Field(
        description="A 2-3 sentence justification for the recommendation, grounded in the resume content"
    )


class CandidateComparison(
    BaseModel
):
    """Structured, ranked comparison across multiple already-evaluated candidates."""
    ranking: List[str] = Field(description="Candidate names/file names, ordered from best fit to worst fit for the role")
    best_candidate: str = Field(description="Name/file name of the single best-fit candidate")
    justification: str = Field(
        description="Explanation of why the best candidate was chosen over the others, "
        "grounded only in the evaluation data provided"
    )


# --- 2. Resume ingestion (Copied from previous cells) ---
class ResumeStore:
    """
    Holds one isolated FAISS vector store per uploaded resume, so retrieval
    for candidate X never leaks context from candidate Y.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.resumes: Dict[str, dict] = {}

    def add_resume_from_path(self, pdf_path: str, candidate_id: str) -> str:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return self._index_pages(pages, candidate_id)

    def add_resume_from_bytes(self, file_bytes: bytes, candidate_id: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
        finally:
            os.unlink(tmp_path)
        return self._index_pages(pages, candidate_id)

    def _index_pages(self, pages: List[Document], candidate_id: str) -> str:
        if not pages:
            raise ValueError(f"No extractable text found in resume '{candidate_id}'.")

        for page in pages:
            page.metadata["source"] = candidate_id

        chunks = self.splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        full_text = "\n".join(p.page_content for p in pages)

        self.resumes[candidate_id] = {
            "vectorstore": vectorstore,
            "chunks": chunks,
            "full_text": full_text,
        }
        return candidate_id

    def get_retriever(self, candidate_id: str, k: int = 6):
        if candidate_id not in self.resumes:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        return self.resumes[candidate_id]["vectorstore"].as_retriever(search_kwargs={"k": k})

    def list_candidates(self) -> List[str]:
        return list(self.resumes.keys())


# --- 3. LLM factory (Copied from previous cells) ---
def build_llm(groq_api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(groq_api_key=groq_api_key, model_name=model, temperature=temperature)


# --- 4. Prompt templates (Copied from previous cells) ---
EVALUATION_PROMPT = PromptTemplate(
    template="""You are an expert technical recruiter and resume screening assistant.\n\nEvaluate the candidate STRICTLY using the retrieved resume excerpts below. These excerpts\nwere retrieved via semantic search from the candidate's actual resume (Retrieval-Augmented\nGeneration). Do not invent, assume, or infer facts, employers, dates, or skills that are not\npresent in the excerpts. If a skill required by the job description is not evidenced in the\nexcerpts, treat it as missing.\n\nJOB DESCRIPTION:\n{job_description}\n\nRETRIEVED RESUME EXCERPTS (candidate: {candidate_name}):\n{context}\n\nNow produce a full evaluation of this candidate against the job description above:\na match score (0-100), the matching skills, the missing skills, a short summary,\nstrengths, weaknesses, and a hiring recommendation with justification.\n\n{format_instructions}\n""",
    input_variables=["job_description", "context", "candidate_name"],
)

QA_PROMPT = PromptTemplate(
    template="""You are a resume screening assistant. Answer the recruiter's question using\nONLY the retrieved resume excerpts below. Do not use outside knowledge and do not guess.\nIf the answer cannot be determined from the excerpts, respond exactly with:\n"This information is not present in the resume."\n\nRESUME EXCERPTS (candidate: {candidate_name}):\n{context}\n\nQUESTION: {question}\n\nANSWER:""",
    input_variables=["candidate_name", "context", "question"],
)

COMPARISON_PROMPT = PromptTemplate(
    template="""You are an expert recruiter comparing several candidates who were each already\nevaluated against the same job description. Use ONLY the structured evaluation data below\n(each of which was itself grounded in that candidate's resume via RAG).\n\n{evaluations_block}\n\nRank the candidates from best fit to worst fit for the role, name the single best candidate,\nand justify the choice using only the data provided above.\n\n{format_instructions}\n""",
    input_variables=["evaluations_block"],
)


# --- 5. Resilient structured parsing (Copied from previous cells) ---
def _parse_with_retry(raw_output: str, parser: PydanticOutputParser, llm: ChatGroq):
    try:
        return parser.parse(raw_output)
    except Exception as first_error:
        fix_prompt = (
            "The following text was supposed to match this format:\n\n"
            f"{parser.get_format_instructions()}\n\n"
            f"Text to fix:\n{raw_output}\n\n"
            f"Parsing error: {first_error}\n\n"
            "Return ONLY the corrected, valid output matching the format above."
        )
        fixed = llm.invoke(fix_prompt).content
        return parser.parse(fixed)


# --- 6. Retrieval + generation chains (Copied from previous cells) ---
def _retrieve_context(store: ResumeStore, candidate_id: str, query: str, k: int) -> str:
    retriever = store.get_retriever(candidate_id, k=k)
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)


def evaluate_resume(
    store: ResumeStore,
    llm: ChatGroq,
    candidate_id: str,
    job_description: str,
    k: int = 6,
) -> ResumeEvaluation:
    context = _retrieve_context(store, candidate_id, job_description, k)

    parser = PydanticOutputParser(pydantic_object=ResumeEvaluation)
    prompt = EVALUATION_PROMPT.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm

    raw = chain.invoke(
        {
            "job_description": job_description,
            "context": context,
            "candidate_name": candidate_id,
        }
    )
    return _parse_with_retry(raw.content, parser, llm)

def answer_question(
    store: ResumeStore,
    llm: ChatGroq,
    candidate_id: str,
    question: str,
    k: int = 6,
) -> str:
    """Answer a free-text recruiter's question using only that candidate's retrieved resume content."""
    context = _retrieve_context(store, candidate_id, question, k)
    prompt = QA_PROMPT.format(candidate_name=candidate_id, context=context, question=question)
    response = llm.invoke(prompt)
    return response.content


def compare_candidates(
    llm: ChatGroq,
    evaluations: Dict[str, ResumeEvaluation],
) -> CandidateComparison:
    if len(evaluations) < 2:
        raise ValueError("Need at least two evaluated candidates to compare.")

    blocks = []
    for name, ev in evaluations.items():
        blocks.append(
            f"""Candidate: {name}\nMatch Score: {ev.match_score}\nMatching Skills: {', '.join(ev.matching_skills) or 'None'}\nMissing Skills: {', '.join(ev.missing_skills) or 'None'}\nStrengths: {', '.join(ev.strengths) or 'None'}\nWeaknesses: {', '.join(ev.weaknesses) or 'None'}\nRecommendation: {ev.recommendation}"""
        )
    evaluations_block = "\n\n".join(blocks)

    parser = PydanticOutputParser(pydantic_object=CandidateComparison)
    prompt = COMPARISON_PROMPT.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm

    raw = chain.invoke({"evaluations_block": evaluations_block})
    return _parse_with_retry(raw.content, parser, llm)


# --- Streamlit Application ---
st.set_page_config(page_title="AI Resume Screener & Comparator", layout="wide")
st.title("🤖 AI-Powered Resume Screener & Comparator")

# --- API Key Setup ---
# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please ensure it's set in a .env file or as an environment variable.")
    st.stop()

# Initialize session state variables
if 'resume_store' not in st.session_state:
    st.session_state.resume_store = ResumeStore()
if 'llm' not in st.session_state:
    st.session_state.llm = build_llm(groq_api_key=GROQ_API_KEY)
if 'uploaded_resumes' not in st.session_state:
    st.session_state.uploaded_resumes = {}

llm = st.session_state.llm
resume_store = st.session_state.resume_store

# --- Sidebar for Uploads ---
st.sidebar.header("Upload Resumes (PDFs)")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        # Only process if not already processed
        if file_name not in st.session_state.uploaded_resumes:
            with st.spinner(f"Processing {file_name}..."):
                try:
                    resume_store.add_resume_from_bytes(uploaded_file.getvalue(), file_name)
                    st.session_state.uploaded_resumes[file_name] = "processed"
                    st.sidebar.success(f"Processed: {file_name}")
                except Exception as e:
                    st.sidebar.error(f"Error processing {file_name}: {e}")

# Display current candidates
st.sidebar.subheader("Loaded Resumes:")
if resume_store.list_candidates():
    for candidate_id in resume_store.list_candidates():
        st.sidebar.write(f"- {candidate_id}")
else:
    st.sidebar.info("No resumes loaded yet.")

# --- Main Content Area ---
st.header("Job Description")
job_description = st.text_area(
    "Paste the job description here",
    height=300,
    value="""
Job Title: Senior Software Engineer

Responsibilities:
- Design, develop, and maintain high-performance web applications.
- Work with Python, Django, and modern JavaScript frameworks like React.
- Manage and optimize PostgreSQL databases.
- Deploy and monitor applications on AWS.

Qualifications:
- 5+ years of experience in software development.
- Expertise in Python, Django, and React.
- Strong understanding of SQL and database optimization.
- Experience with cloud services, especially AWS.
- Excellent problem-solving abilities.
"""
)

if st.button("Evaluate and Compare Resumes"):
    if not job_description:
        st.warning("Please provide a job description.")
    elif not resume_store.list_candidates():
        st.warning("Please upload at least one resume.")
    else:
        st.subheader("Individual Resume Evaluations")
        evaluated_results = {}
        for candidate_id in resume_store.list_candidates():
            with st.spinner(f"Evaluating {candidate_id}..."):
                try:
                    evaluation = evaluate_resume(
                        store=resume_store,
                        llm=llm,
                        candidate_id=candidate_id,
                        job_description=job_description
                    )
                    evaluated_results[candidate_id] = evaluation
                    st.success(f"Evaluation complete for {candidate_id}")

                    st.write(f"### {evaluation.candidate_name}")
                    st.write(f"**Match Score:** {evaluation.match_score}/100")
                    st.write(f"**Recommendation:** {evaluation.recommendation}")
                    st.write(f"**Summary:** {evaluation.summary}")
                    with st.expander("Details"):
                        st.json(evaluation.model_dump_json(indent=2))

                except Exception as e:
                    st.error(f"Error evaluating {candidate_id}: {e}")

        if len(evaluated_results) >= 2:
            st.subheader("Candidate Comparison")
            with st.spinner("Comparing candidates..."):
                try:
                    comparison = compare_candidates(
                        llm=llm,
                        evaluations=evaluated_results
                    )
                    st.success("Comparison complete!")
                    st.write(f"### Best Candidate: {comparison.best_candidate}")
                    st.write(f"**Ranking:** {', '.join(comparison.ranking)}")
                    st.write(f"**Justification:** {comparison.justification}")
                    with st.expander("Raw Comparison Data"):
                        st.json(comparison.model_dump_json(indent=2))
                except Exception as e:
                    st.error(f"Error comparing candidates: {e}")
        elif len(evaluated_results) == 1:
            st.info("Upload more resumes to enable candidate comparison.")
        else:
            st.warning("No resumes were successfully evaluated for comparison.")
