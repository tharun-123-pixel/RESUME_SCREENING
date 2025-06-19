<<<<<<< HEAD
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Define category-specific skills
CATEGORY_SKILLS = {
    "Data Science": [
        "python", "sql", "r", "pandas", "numpy", "scikit-learn", "polars", "tensorflow", "pytorch",
        "plotly", "matplotlib", "seaborn", "tableau", "power bi", "spark", "hadoop", "aws", "gcp", "azure",
        "postgresql", "mysql", "mongodb", "dynamodb"
    ],
    "Data Analyst": [
        "sql", "python", "pandas", "excel", "tableau", "power bi", "google data studio", "mysql",
        "postgresql", "snowflake", "looker studio", "power apps", "qlik sense", "openrefine", "alteryx"
    ],
    "Software Development": [
        "python", "javascript", "typescript", "java", "c#", "go", "react", "vue.js", "node.js",
        "spring boot", "fastapi", "flask", "git", "postgresql", "mysql", "mongodb", "redis",
        "docker", "kubernetes", "rest api", "graphql", "jest", "pytest"
    ],
    "UI/UX Designer": [
        "figma", "adobe xd", "sketch", "canva", "invision", "surveymonkey", "google forms", "maze",
        "html", "css", "javascript", "lottie", "framer"
    ],
    "AI/ML": [
        "python", "tensorflow", "pytorch", "scikit-learn", "jax", "keras", "opencv", "yolo",
        "spark", "aws sagemaker", "gcp vertex ai", "azure ml", "mlflow", "tfx", "fastapi", "flask",
        "aws", "gcp", "azure", "sql", "airflow", "prefect"
    ]
}

# Synthetic dataset
data = {
    'resume_text': [
        # Data Science
        "Skilled in python pandas numpy sql tensorflow pytorch matplotlib seaborn tableau aws spark",
        "Experienced in r sql data visualization pandas scikit-learn polars gcp mysql mongodb",
        "Proficient in python sql numpy matplotlib power bi hadoop postgresql dynamodb",
        # Data Analyst
        "Expert in sql python pandas excel tableau power bi mysql snowflake looker studio",
        "Skilled in sql excel google data studio postgresql openrefine power apps analytics",
        "Proficient in sql python tableau power bi qlik sense alteryx data cleaning",
        # Software Development
        "Developed apps using javascript react node.js mongodb fastapi git docker rest api",
        "Full-stack developer with python java spring boot postgresql redis kubernetes pytest",
        "Proficient in typescript vue.js node.js graphql mysql jest ci/cd pipelines",
        # UI/UX Designer
        "Designed interfaces with figma adobe xd html css surveymonkey maze prototyping",
        "Skilled in sketch canva invision lottie framer user research accessibility",
        "Proficient in figma html css javascript google forms ui design animations",
        # AI/ML
        "Expert in python tensorflow pytorch scikit-learn nlp computer vision aws sagemaker",
        "Skilled in python keras jax opencv mlflow fastapi gcp vertex ai spark",
        "Proficient in tensorflow pytorch yolo azure ml airflow prefect sql cloud"
    ],
    'category': [
        "Data Science", "Data Science", "Data Science",
        "Data Analyst", "Data Analyst", "Data Analyst",
        "Software Development", "Software Development", "Software Development",
        "UI/UX Designer", "UI/UX Designer", "UI/UX Designer",
        "AI/ML", "AI/ML", "AI/ML"
    ]
}

df = pd.DataFrame(data)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['resume_text'], y, test_size=0.2, random_state=42)

# Create and train model
tfidf = TfidfVectorizer(stop_words='english')
clf = LogisticRegression(max_iter=1000)
model = make_pipeline(tfidf, clf)
model.fit(X_train, y_train)

# Save model, encoder, and skills
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('category_skills.pkl', 'wb') as f:
    pickle.dump(CATEGORY_SKILLS, f)

=======
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Define category-specific skills
CATEGORY_SKILLS = {
    "Data Science": [
        "python", "sql", "r", "pandas", "numpy", "scikit-learn", "polars", "tensorflow", "pytorch",
        "plotly", "matplotlib", "seaborn", "tableau", "power bi", "spark", "hadoop", "aws", "gcp", "azure",
        "postgresql", "mysql", "mongodb", "dynamodb"
    ],
    "Data Analyst": [
        "sql", "python", "pandas", "excel", "tableau", "power bi", "google data studio", "mysql",
        "postgresql", "snowflake", "looker studio", "power apps", "qlik sense", "openrefine", "alteryx"
    ],
    "Software Development": [
        "python", "javascript", "typescript", "java", "c#", "go", "react", "vue.js", "node.js",
        "spring boot", "fastapi", "flask", "git", "postgresql", "mysql", "mongodb", "redis",
        "docker", "kubernetes", "rest api", "graphql", "jest", "pytest"
    ],
    "UI/UX Designer": [
        "figma", "adobe xd", "sketch", "canva", "invision", "surveymonkey", "google forms", "maze",
        "html", "css", "javascript", "lottie", "framer"
    ],
    "AI/ML": [
        "python", "tensorflow", "pytorch", "scikit-learn", "jax", "keras", "opencv", "yolo",
        "spark", "aws sagemaker", "gcp vertex ai", "azure ml", "mlflow", "tfx", "fastapi", "flask",
        "aws", "gcp", "azure", "sql", "airflow", "prefect"
    ]
}

# Synthetic dataset
data = {
    'resume_text': [
        # Data Science
        "Skilled in python pandas numpy sql tensorflow pytorch matplotlib seaborn tableau aws spark",
        "Experienced in r sql data visualization pandas scikit-learn polars gcp mysql mongodb",
        "Proficient in python sql numpy matplotlib power bi hadoop postgresql dynamodb",
        # Data Analyst
        "Expert in sql python pandas excel tableau power bi mysql snowflake looker studio",
        "Skilled in sql excel google data studio postgresql openrefine power apps analytics",
        "Proficient in sql python tableau power bi qlik sense alteryx data cleaning",
        # Software Development
        "Developed apps using javascript react node.js mongodb fastapi git docker rest api",
        "Full-stack developer with python java spring boot postgresql redis kubernetes pytest",
        "Proficient in typescript vue.js node.js graphql mysql jest ci/cd pipelines",
        # UI/UX Designer
        "Designed interfaces with figma adobe xd html css surveymonkey maze prototyping",
        "Skilled in sketch canva invision lottie framer user research accessibility",
        "Proficient in figma html css javascript google forms ui design animations",
        # AI/ML
        "Expert in python tensorflow pytorch scikit-learn nlp computer vision aws sagemaker",
        "Skilled in python keras jax opencv mlflow fastapi gcp vertex ai spark",
        "Proficient in tensorflow pytorch yolo azure ml airflow prefect sql cloud"
    ],
    'category': [
        "Data Science", "Data Science", "Data Science",
        "Data Analyst", "Data Analyst", "Data Analyst",
        "Software Development", "Software Development", "Software Development",
        "UI/UX Designer", "UI/UX Designer", "UI/UX Designer",
        "AI/ML", "AI/ML", "AI/ML"
    ]
}

df = pd.DataFrame(data)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['resume_text'], y, test_size=0.2, random_state=42)

# Create and train model
tfidf = TfidfVectorizer(stop_words='english')
clf = LogisticRegression(max_iter=1000)
model = make_pipeline(tfidf, clf)
model.fit(X_train, y_train)

# Save model, encoder, and skills
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('category_skills.pkl', 'wb') as f:
    pickle.dump(CATEGORY_SKILLS, f)

>>>>>>> 85c488e4310da85196688891b8d7f8435f1da17f
print("Model and resources saved successfully!")