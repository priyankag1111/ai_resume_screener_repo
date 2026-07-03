
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Ensure the output directory exists
OUTPUT_DIR = "/content/sample_resumes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_pdf(filepath, content):
    c = canvas.Canvas(filepath, pagesize=letter)
    textobject = c.beginText()
    textobject.setTextOrigin(50, 750) # Start from top-leftish
    for line in content.split('\n'):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()
    print(f"Generated: {filepath}")

# Resume A: Priya Sharma (Strong Data Scientist)
resume_a_content = """
Priya Sharma
Data Scientist

Summary:
Experienced Data Scientist with 5+ years in developing and deploying machine learning models, statistical analysis, and A/B testing. Proficient in Python, SQL, and cloud platforms like AWS SageMaker.

Experience:
- Senior Data Scientist at TechCorp (3 years): Led projects in predictive modeling, utilized Python (scikit-learn, TensorFlow), SQL, and deployed models on AWS SageMaker. Designed and analyzed A/B tests.
- Data Scientist at InnovateX (2 years): Developed data pipelines, performed statistical analysis, and built dashboards with Tableau.

Skills: Python, SQL, Machine Learning (scikit-learn, TensorFlow, XGBoost), AWS SageMaker, A/B Testing, Statistical Analysis, Tableau, Data Visualization, ETL
Education: M.Sc. Data Science, University of X
"""
create_pdf(os.path.join(OUTPUT_DIR, "Resume_A_Priya_Sharma.pdf"), resume_a_content)

# Resume B: James Whitfield (Software Engineer transitioning to Data Science)
resume_b_content = """
James Whitfield
Software Engineer

Summary:
Software Engineer with 4 years of experience in Python and software development. Eager to transition into Data Science, with foundational knowledge in machine learning and data analysis.

Experience:
- Software Engineer at CodeWorks (4 years): Developed backend services using Python and Django. Worked with PostgreSQL databases. Participated in some internal data analysis projects.

Skills: Python, Django, PostgreSQL, Git, JavaScript, Basic Machine Learning Concepts, Data Cleaning
Education: B.Sc. Computer Science, University of Y
"""
create_pdf(os.path.join(OUTPUT_DIR, "Resume_B_James_Whitfield.pdf"), resume_b_content)

# Resume C: Diego Alvarez (Business Analyst with some data skills)
resume_c_content = """
Diego Alvarez
Business Analyst

Summary:
Business Analyst with 6 years of experience in market research, data interpretation, and stakeholder communication. Has worked with Excel and some basic SQL for reporting.

Experience:
- Senior Business Analyst at Global Insights (4 years): Analyzed market trends, prepared reports, and presented findings. Used advanced Excel and Power BI.
- Business Analyst at ConsultCo (2 years): Supported client projects with data collection and basic statistical reporting.

Skills: Market Research, Data Interpretation, SQL (basic), Excel, Power BI, Communication, Project Management
Education: B.A. Business Administration, University of Z
"""
create_pdf(os.path.join(OUTPUT_DIR, "Resume_C_Diego_Alvarez.pdf"), resume_c_content)
