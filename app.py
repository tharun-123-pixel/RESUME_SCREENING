import streamlit as st
import pdfplumber
import re
import spacy
import pickle
from datetime import datetime

# Cache SpaCy model for performance
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Failed to load SpaCy model: {e}")
        raise e

try:
    nlp = load_spacy_model()
except Exception:
    st.stop()

# Load pre-trained model and resources
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('category_skills.pkl', 'rb') as f:
        CATEGORY_SKILLS = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load pickle files: {e}. Ensure model.pkl, encoder.pkl, and category_skills.pkl are in the same directory.")
    st.stop()

# Update CATEGORY_SKILLS with new categories
CATEGORY_SKILLS.update({
    "Frontend Developer": [
        "HTML", "CSS", "JavaScript", "TypeScript", "React", "Next.js", "Vue.js",
        "Tailwind CSS", "GSAP", "Framer Motion", "Figma", "Jest"
    ],
    "Backend Developer": [
        "Node.js", "Express.js", "Python", "Java", "Go", "Spring Boot", "FastAPI",
        "Flask", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Docker",
        "Kubernetes", "REST API", "GraphQL", "Pytest"
    ],
    "Full Stack Developer": [
        "HTML", "CSS", "JavaScript", "TypeScript", "React", "Next.js", "Vue.js",
        "Tailwind CSS", "GSAP", "Framer Motion", "Figma", "Jest",
        "Node.js", "Express.js", "Python", "Java", "Go", "Spring Boot", "FastAPI",
        "Flask", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Docker",
        "Kubernetes", "REST API", "GraphQL", "Pytest", "Git", "AWS", "Azure",
        "Vercel", "Netlify", "Zustand"
    ]
})

# Flatten CATEGORY_SKILLS and include variations + academic terms
ALL_SKILLS = set()
for skills in CATEGORY_SKILLS.values():
    ALL_SKILLS.update(skill.lower() for skill in skills)
ALL_SKILLS.update([
    'html5', 'css3', 'react.js', 'express.js', 'framer motion', 'next.js', 
    'tailwind css', 'gsap', 'vercel', 'netlify', 'zustand', 
    'algorithms', 'data structures', 'software engineering'
])

# Learning resources for skills
LEARNING_RESOURCES = {
    "python": ["https://www.codecademy.com/learn/learn-python", "https://www.coursera.org/learn/python"],
    "sql": ["https://www.w3schools.com/sql/", "https://www.sqlzoo.net/"],
    "pandas": ["https://pandas.pydata.org/docs/getting_started/index.html"],
    "numpy": ["https://numpy.org/learn/"],
    "scikit-learn": ["https://scikit-learn.org/stable/tutorial/"],
    "pytorch": ["https://pytorch.org/tutorials/"],
    "tensorflow": ["https://www.tensorflow.org/learn"],
    "matplotlib": ["https://matplotlib.org/stable/users/index.html"],
    "seaborn": ["https://seaborn.pydata.org/"],
    "tableau": ["https://www.tableau.com/learn"],
    "power bi": ["https://learn.microsoft.com/en-us/power-bi/"],
    "spark": ["https://spark.apache.org/docs/latest/"],
    "hadoop": ["https://hadoop.apache.org/docs/stable/"],
    "aws": ["https://aws.amazon.com/training/"],
    "gcp": ["https://cloud.google.com/learn"],
    "azure": ["https://learn.microsoft.com/en-us/azure/"],
    "postgresql": ["https://www.postgresqltutorial.com/"],
    "mysql": ["https://dev.mysql.com/doc/"],
    "mongodb": ["https://www.mongodb.com/docs/"],
    "dynamodb": ["https://aws.amazon.com/dynamodb/"],
    "excel": ["https://support.microsoft.com/excel"],
    "javascript": ["https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"],
    "typescript": ["https://www.typescriptlang.org/docs/"],
    "java": ["https://www.oracle.com/java/technologies/"],
    "c#": ["https://learn.microsoft.com/en-us/dotnet/csharp/"],
    "go": ["https://go.dev/learn/"],
    "react": ["https://react.dev/learn"],
    "vue.js": ["https://vuejs.org/guide/"],
    "node.js": ["https://nodejs.org/en/learn"],
    "spring boot": ["https://spring.io/projects/spring-boot#learn"],
    "fastapi": ["https://fastapi.tiangolo.com/tutorial/"],
    "flask": ["https://flask.palletsprojects.com/en/stable/"],
    "git": ["https://git-scm.com/doc"],
    "redis": ["https://redis.io/docs/"],
    "docker": ["https://docs.docker.com/get-started/"],
    "kubernetes": ["https://kubernetes.io/docs/"],
    "rest api": ["https://restfulapi.net/"],
    "graphql": ["https://graphql.org/learn/"],
    "jest": ["https://jestjs.io/docs/"],
    "pytest": ["https://docs.pytest.org/en/stable/"],
    "figma": ["https://www.figma.com/resources/learn-design/"],
    "adobe xd": ["https://helpx.adobe.com/xd/get-started.html"],
    "html": ["https://www.w3schools.com/html/"],
    "css": ["https://developer.mozilla.org/en-US/docs/Web/CSS"],
    "lottie": ["https://lottiefiles.com/learn"],
    "framer": ["https://www.framer.com/learn/"],
    "framer motion": ["https://www.framer.com/motion/"],
    "jax": ["https://jax.readthedocs.io/en/stable/"],
    "keras": ["https://keras.io/guides/"],
    "opencv": ["https://opencv.org/get-started/"],
    "yolo": ["https://docs.ultralytics.com/"],
    "mlflow": ["https://mlflow.org/docs/"],
    "tfx": ["https://www.tensorflow.org/tfx"],
    "airflow": ["https://airflow.apache.org/docs/"],
    "prefect": ["https://docs.prefect.io/"],
    "next.js": ["https://nextjs.org/learn"],
    "tailwind css": ["https://tailwindcss.com/docs"],
    "gsap": ["https://greensock.com/docs/"],
    "vercel": ["https://vercel.com/docs"],
    "netlify": ["https://docs.netlify.com/"],
    "zustand": ["https://zustand-demo.pmnd.rs/"],
    "express.js": ["https://expressjs.com/en/starter/installing.html"]
}

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text:
            st.error("No text extracted from PDF. Ensure it's a text-based PDF.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}. Ensure the PDF is valid and not scanned.")
        return ""

def extract_name(text, skills_set=ALL_SKILLS):
    """Extract name using regex, prioritizing multi-word names, with SpaCy fallback."""
    try:
        text = re.sub(r'[\s\r\n]+', ' ', text.strip())
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name_pattern = re.compile(r'^[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*$')
        if lines:
            first_line = lines[0]
            if name_pattern.match(first_line):
                if skills_set is None or first_line.lower() not in skills_set:
                    return first_line
        if len(lines) > 1:
            second_line = lines[1]
            if name_pattern.match(second_line) and (skills_set is None or second_line.lower() not in skills_set):
                return second_line
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and (skills_set is None or ent.text.lower() not in skills_set):
                return ent.text.strip()
        return "Unknown"
    except Exception as e:
        st.error(f"Error extracting name: {e}")
        return "Unknown"

def extract_birth_year(text):
    """Extract birth year and calculate age."""
    try:
        birth_year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        match = birth_year_pattern.search(text)
        if match:
            birth_year = int(match.group())
            current_year = datetime.now().year
            age = current_year - birth_year
            if 15 <= age <= 100:
                return age
        return None
    except Exception as e:
        st.error(f"Error extracting birth year: {e}")
        return None

def extract_email(text):
    """Extract email address."""
    try:
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        match = email_pattern.search(text)
        return match.group() if match else "None"
    except Exception as e:
        st.error(f"Error extracting email: {e}")
        return "None"

def extract_phone_number(text):
    """Extract phone number."""
    try:
        phone_pattern = re.compile(r'\b(?:\+?1\s*?)?(?:\(\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b')
        match = phone_pattern.search(text)
        return match.group() if match else "None"
    except Exception as e:
        st.error(f"Error extracting phone number: {e}")
        return "None"

def extract_skills(text, target_skills):
    """Extract skills from resume matching target skills with variation handling."""
    try:
        found_skills = set()
        text_lower = text.lower()
        skill_mappings = {
            "react.js": "react",
            "html5": "html",
            "css3": "css",
            "framer motion": "framer motion",
            "next.js": "next.js",
            "tailwind css": "tailwind css",
            "gsap": "gsap",
            "vercel": "vercel",
            "netlify": "netlify",
            "zustand": "zustand",
            "express.js": "express.js"
        }
        for skill in target_skills:
            patterns = [r'\b' + re.escape(skill.lower()).replace(' ', r'\s*(?:,\s*|\s+|/)') + r'\b']
            if skill.lower() in skill_mappings.values():
                for key, value in skill_mappings.items():
                    if value == skill.lower():
                        patterns.append(r'\b' + re.escape(key.lower()).replace(' ', r'\s*(?:,\s*|\s+|/)') + r'\b')
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found_skills.add(skill.lower())
                    break
        return sorted(list(found_skills))
    except Exception as e:
        st.error(f"Error extracting skills: {e}")
        return []

def clean_resume(resume_str):
    """Clean resume text by removing sensitive data."""
    try:
        resume_str = re.sub(r'https?://\S+', '', resume_str)
        resume_str = re.sub(r'\S+@\S+\.\S+', '', resume_str)
        resume_str = re.sub(r'\b\d{10}\b', '', resume_str)
        return resume_str
    except Exception as e:
        st.error(f"Error cleaning resume: {e}")
        return resume_str

def predict_category(resume_text):
    """Predict job category using pre-trained model."""
    try:
        resume_cleaned = clean_resume(resume_text)
        y_pred = model.predict([resume_cleaned])
        return encoder.inverse_transform(y_pred)[0]
    except Exception as e:
        st.error(f"Error predicting category with model: {e}")
        return "Unknown"

def calculate_resume_score(extracted_skills, target_skills):
    """Calculate a resume score based on matched skills."""
    try:
        if not target_skills:
            return 0
        matched_skills = len(extracted_skills)
        total_skills = len(target_skills)
        return round((matched_skills / total_skills) * 100, 2)
    except Exception as e:
        st.error(f"Error calculating resume score: {e}")
        return 0

def find_best_category(resume_text, category_skills):
    """Find the category with the highest skill match score."""
    try:
        best_category = None
        best_score = -1
        best_extracted_skills = []
        best_target_skills = []

        for category, skills in category_skills.items():
            extracted_skills = extract_skills(resume_text, skills)
            score = calculate_resume_score(extracted_skills, skills)
            # st.write(f"DEBUG: Category {category} scored {score}%")  # Debug logging
            if score > best_score:
                best_category = category
                best_score = score
                best_extracted_skills = extracted_skills
                best_target_skills = skills

        if best_category is None:
            # Fallback to model prediction if no skills match
            best_category = predict_category(resume_text)
            best_target_skills = category_skills.get(best_category, [])
            best_extracted_skills = extract_skills(resume_text, best_target_skills)
            best_score = calculate_resume_score(best_extracted_skills, best_target_skills)
            # st.write(f"DEBUG: Fallback to model prediction: {best_category} with score {best_score}%")

        return best_category, best_score, best_extracted_skills, best_target_skills
    except Exception as e:
        st.error(f"Error finding best category: {e}")
        return "Unknown", 0, [], []

def predict_selection_chance(score, extracted_skills, target_skills, resume_text, selected_category, category_skills):
    """Predict selection chance and provide improvement and safer side recommendations."""
    try:
        # st.write(f"DEBUG: predict_selection_chance called with category: {selected_category}, score: {score}%")  # Debug logging

        # Define selection chance
        if score > 80:
            chance = "High"
            chance_desc = "Your resume closely matches the requirements, giving a strong chance of selection."
        elif score >= 50:
            chance = "Moderate"
            chance_desc = "Your resume is competitive but needs more skills to improve selection odds."
        else:
            chance = "Low"
            chance_desc = "Your resume lacks critical skills, reducing selection chances."

        # Improvement recommendations
        missing_skills = [skill for skill in target_skills if skill.lower() not in extracted_skills]
        improvement_tips = []
        if missing_skills:
            improvement_tips.append(f"Learn these missing skills: {', '.join(missing_skills)}.")
            improvement_tips.append("Add projects or certifications showcasing these skills.")
        if extracted_skills:
            improvement_tips.append(f"Highlight skills ({', '.join(extracted_skills)}) in your resume’s skills section or projects.")
        improvement_tips.append("Use ATS-friendly formats with clear headings and job-relevant keywords.")

        # Safer side strategy
        alternative_categories = []
        for category, skills in category_skills.items():
            if category != selected_category:
                alt_extracted_skills = extract_skills(resume_text, skills)
                alt_score = calculate_resume_score(alt_extracted_skills, skills)
                # st.write(f"DEBUG: Alternative category {category} scored {alt_score}%")  # Debug logging
                alternative_categories.append((category, alt_score))
        alternative_categories = sorted(alternative_categories, key=lambda x: x[1], reverse=True)[:2]

        safer_side = []
        if chance in ["Moderate", "Low"]:
            if alternative_categories and any(alt_score > score for _, alt_score in alternative_categories):
                safer_side.append(f"Consider these roles with better skill alignment (based on: {selected_category}):")
                for alt_cat, alt_score in alternative_categories:
                    if alt_score > score:
                        safer_side.append(f"- {alt_cat} (Score: {alt_score}%)")
            else:
                safer_side.append(f"No better-matching roles found. Focus on improving skills for {selected_category}.")
            safer_side.append(f"Apply for entry-level or internships in {selected_category} or related fields.")
            safer_side.append("Network with industry professionals to boost visibility.")
        else:
            safer_side.append(f"Your skills align well with {selected_category}; tailor your resume to job postings.")

        return {
            "chance": chance,
            "chance_description": chance_desc,
            "improvement_tips": improvement_tips,
            "safer_side": safer_side
        }
    except Exception as e:
        st.error(f"Error predicting selection chance: {e}")
        return {
            "chance": "Unknown",
            "chance_description": "Unable to predict selection chance due to an error.",
            "improvement_tips": ["Resolve the error and try again."],
            "safer_side": ["Resolve the error and try again."]
        }

# Streamlit App
st.title("📄 Resume Analyzer")
st.write("Select a job category or let the model predict one, then upload your resume.")

# Category selection
try:
    category_option = st.selectbox(
        "Select Job Category (or 'Predict' for auto-detection)",
        options=["Predict", "Data Science", "Data Analyst", "Software Development",
                 "UI/UX Designer", "AI/ML", "Frontend Developer",
                 "Backend Developer", "Full Stack Developer"]
    )
except Exception as e:
    st.error(f"Error rendering category selector: {e}")
    st.stop()

st.write("📤 Upload your resume (PDF only)")
try:
    uploaded_file = st.file_uploader("", type=["pdf"], accept_multiple_files=False)
except Exception as e:
    st.error(f"Error rendering file uploader: {e}")
    st.stop()

if uploaded_file:
    with st.spinner("Processing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            st.success("PDF uploaded and processed successfully!")

            # Extract details
            name = extract_name(resume_text, ALL_SKILLS)
            age = extract_birth_year(resume_text)
            email = extract_email(resume_text)
            phone = extract_phone_number(resume_text)

            # Determine category and skills
            if category_option == "Predict":
                category, score, extracted_skills, target_skills = find_best_category(resume_text, CATEGORY_SKILLS)
                # st.write(f"DEBUG: Predicted category: {category} with score {score}%")  # Debug logging
            else:
                category = category_option
                target_skills = CATEGORY_SKILLS.get(category, [])
                extracted_skills = extract_skills(resume_text, target_skills)
                score = calculate_resume_score(extracted_skills, target_skills)
                # st.write(f"DEBUG: Selected category: {category} with score {score}%")  # Debug logging

            # Predict selection chance
            selection_info = predict_selection_chance(score, extracted_skills, target_skills, resume_text, category, CATEGORY_SKILLS)

            # Display results
            st.header("📊 Analysis Result")
            st.write(f"🧑 **Name**: {name}")
            st.write(f"📧 **Email**: {email}")
            st.write(f"📞 **Phone Number**: {phone}")
            if age:
                st.write(f"🎂 **Estimated Age**: {age}")
            st.write(f"💼 **Category**: {category} {'(Predicted)' if category_option == 'Predict' else '(Selected)'}")
            st.write(f"📌 **Extracted Skills**: {', '.join(extracted_skills) if extracted_skills else 'None'}")
            st.write(f"🏆 **Resume Score**: {score}% (based on {len(extracted_skills)} out of {len(target_skills)} required skills)")
            st.progress(score / 100)
            st.write(f"🎯 **Selection Chance**: {selection_info['chance']} ({selection_info['chance_description']})")

            # Display recommendations
            st.subheader("🚀 Recommendations")
            st.write("**To Improve Your Resume**:")
            for tip in selection_info['improvement_tips']:
                st.write(f"- {tip}")
            st.write("**Safer Side Strategy**:")
            for strategy in selection_info['safer_side']:
                st.write(f"- {strategy}")

            # Learning resources
            missing_skills = [skill for skill in target_skills if skill.lower() not in extracted_skills]
            if missing_skills:
                st.subheader("📚 Recommended Learning Resources")
                seen_urls = set()
                for skill in missing_skills:
                    resources = LEARNING_RESOURCES.get(skill.lower(), [])
                    for url in resources:
                        if url not in seen_urls:
                            st.markdown(f"🔗 [Learn {skill.title()}]({url})")
                            seen_urls.add(url)
            else:
                st.write(f"✅ **Feedback**: Great! You have all the necessary skills for {category}.")
        else:
            st.error("Failed to extract text from the PDF. Ensure it's a text-based PDF and try again.")