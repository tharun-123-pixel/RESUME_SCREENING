# 📄 Resume Analyzer

**Resume Analyzer** is a web application built using **Python** and **Streamlit** that helps users evaluate their resumes and understand how well they match specific job roles. Users can upload a **PDF resume**, and the app instantly extracts key details like **name, email, phone number, age**, and **technical skills** using **SpaCy** and regular expressions.

The app uses a **machine learning model** (Logistic Regression with TF-IDF) to either **predict the most suitable job category** or analyze a user-selected one. It compares the extracted skills with role-specific requirements and calculates a **resume score**, indicating how well the resume aligns with the chosen role. Based on the score, the app predicts the **chance of selection** as High, Moderate, or Low.

It also provides **personalized recommendations**, such as missing skills, resume improvement tips, and curated **learning resources** from trusted platforms like Coursera, Codecademy, and W3Schools. If the selected role isn’t the best fit, the app suggests **alternative job categories** based on better skill alignment.

All machine learning resources—including the model, encoder, and skill mappings—are stored using `pickle`. The app is simple, interactive, and ideal for **students, job seekers**, or **early-career professionals** preparing for tech roles like **Data Scientist, Software Developer, UI/UX Designer**, and more.
