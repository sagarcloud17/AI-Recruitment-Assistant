# app.py
import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import json
from PyPDF2 import PdfReader
from openai import OpenAI
import time
from dotenv import load_dotenv
import re
from collections import Counter

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with error handling
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client. Please check your API key: {str(e)}")
    client = None

# Set page configuration
st.set_page_config(
    page_title="AI Recruitment Assistant",
    page_icon="👔",
    layout="wide"
)

# App title and description
st.title("AI Recruitment Assistant")
st.markdown("""
This application helps you match job descriptions with candidate profiles to find the best matches.

**How it works:**
1. Upload a job description (PDF)
2. Upload candidate profiles (CSV or JSON)
3. View ranked matches with detailed explanations
""")

# Cache for embeddings to avoid redundant API calls
embedding_cache = {}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to get embeddings from OpenAI with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_embedding(text, model="text-embedding-3-small"):
    # Create a cache key
    cache_key = f"{text[:100]}_{model}"
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    try:
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return np.zeros(1536)  # Default dimension for text-embedding-3-small

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
    
    # Avoid division by zero
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
        
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

# Function to extract keywords from text
def extract_keywords(text):
    # Remove special characters and split into words
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9+#.\-_]*\b', text.lower())
    # Remove common stop words
    stop_words = {"and", "the", "of", "to", "in", "a", "for", "with", "on", "at", "by", "an", "is", "are", "from", "or", "this", "that"}
    words = [word for word in words if word not in stop_words and len(word) > 1]
    # Count word frequencies
    return Counter(words)

# Function to extract key requirements from job description using GPT-4
def extract_requirements(job_description):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in HR and recruitment. Extract the key skills, qualifications, and requirements from the given job description."},
                {"role": "user", "content": f"""Please analyze this job description and extract key requirements:

{job_description}

Format your response as JSON with the following structure:
{{
    "required_skills": [],
    "preferred_skills": [],
    "qualifications": [],
    "experience": [],
    "responsibilities": [],
    "attributes": []
}}

Make sure to extract as many relevant skills, qualifications, and requirements as possible. For each category, list items in order of importance."""}
            ],
            response_format={"type": "json_object"}
        )
        
        requirements = json.loads(response.choices[0].message.content)
        
        # Create keyword sets for each category
        requirements['keyword_sets'] = {
            'required_skills': extract_keywords(' '.join(requirements['required_skills'])),
            'preferred_skills': extract_keywords(' '.join(requirements['preferred_skills'])),
            'qualifications': extract_keywords(' '.join(requirements['qualifications'])),
            'experience': extract_keywords(' '.join(requirements['experience'])),
            'responsibilities': extract_keywords(' '.join(requirements.get('responsibilities', []))),
            'attributes': extract_keywords(' '.join(requirements.get('attributes', [])))
        }
        
        return requirements
    except Exception as e:
        st.error(f"Error extracting requirements: {str(e)}")
        # Return a basic structure if API call fails
        return {
            "required_skills": [],
            "preferred_skills": [],
            "qualifications": [],
            "experience": [],
            "responsibilities": [],
            "attributes": [],
            "keyword_sets": {}
        }

# Function to calculate keyword match score
def calculate_keyword_match(profile, requirements):
    # Extract keywords from profile
    profile_text = f"{profile['current_title']} {' '.join(profile['skills'])} {profile['experience']} {profile['education']}"
    profile_keywords = extract_keywords(profile_text)
    
    # Calculate matches for each category
    scores = {}
    total_weight = 0
    
    # Category weights
    weights = {
        'required_skills': 0.40,  # Most important
        'preferred_skills': 0.20,
        'qualifications': 0.15,
        'experience': 0.15,
        'responsibilities': 0.05,
        'attributes': 0.05
    }
    
    for category, category_keywords in requirements['keyword_sets'].items():
        if not category_keywords:
            continue
            
        # Count matches
        matches = sum((profile_keywords & category_keywords).values())
        possible_matches = sum(category_keywords.values())
        
        # Calculate category score
        if possible_matches > 0:
            category_score = (matches / possible_matches) * 100
        else:
            category_score = 0
            
        scores[category] = category_score
        total_weight += weights.get(category, 0)
    
    # Normalize weights if some categories are missing
    normalized_weights = {k: v/total_weight for k, v in weights.items() if k in scores}
    
    # Calculate weighted average score
    if normalized_weights:
        weighted_score = sum(scores[category] * normalized_weights[category] for category in scores)
    else:
        weighted_score = 0
        
    return {
        'keyword_score': weighted_score,
        'category_scores': scores
    }

# Function to analyze candidate profile against job requirements
def analyze_profile_match(job_description, profile, requirements, keyword_match_results):
    # Format profile details for GPT-4
    profile_text = f"""
Name: {profile['name']}
Current Title: {profile['current_title']}
Skills: {', '.join(profile['skills'])}
Experience: {profile['experience']}
Education: {profile['education']}
"""
    
    # Format requirements for GPT-4
    requirements_text = f"""
Required Skills: {', '.join(requirements['required_skills'])}
Preferred Skills: {', '.join(requirements['preferred_skills'])}
Qualifications: {', '.join(requirements['qualifications'])}
Experience: {', '.join(requirements['experience'])}
"""
    
    # Additional context based on keyword analysis
    keyword_context = f"""
The candidate's keyword match scores are:
- Overall keyword match: {keyword_match_results['keyword_score']:.1f}%
"""
    for category, score in keyword_match_results['category_scores'].items():
        keyword_context += f"- {category.replace('_', ' ').title()}: {score:.1f}%\n"
    
    try:
        # Analyze match using GPT-4
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Using a slightly less expensive model than gpt-4.1
            messages=[
                {"role": "system", "content": "You are an expert recruiter evaluating candidate profiles against job requirements."},
                {"role": "user", "content": f"""
Job Description:
{job_description[:1000]}...  # Truncated to reduce token usage

Job Requirements:
{requirements_text}

Candidate Profile:
{profile_text}

Additional Context:
{keyword_context}

Evaluate this candidate against the job requirements. Consider both the keyword match scores provided and your qualitative assessment. Return a JSON with:
1. match_score (0-100, be realistic and consider that a perfect 100 is virtually impossible)
2. strengths (list of candidate's top 3-5 strengths for this role)
3. gaps (list of top 3-5 missing skills or experience)
4. overall_assessment (2-3 sentence summary)
"""}
            ],
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        # Blend the keyword score with the GPT analysis score (70% GPT, 30% keyword)
        blended_score = 0.7 * analysis['match_score'] + 0.3 * keyword_match_results['keyword_score']
        analysis['match_score'] = round(blended_score)
        
        return analysis
    except Exception as e:
        st.error(f"Error analyzing profile: {str(e)}")
        # Return a default analysis if API call fails
        return {
            'match_score': round(keyword_match_results['keyword_score']),
            'strengths': ["Could not generate detailed strengths due to API error"],
            'gaps': ["Could not generate detailed gaps due to API error"],
            'overall_assessment': f"Based on keyword matching only, this candidate has a {keyword_match_results['keyword_score']:.1f}% match with the job requirements."
        }

# Function to load profiles from file
def load_profiles(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            
            # Convert string skills to lists if they're comma-separated
            if 'skills' in df.columns and isinstance(df['skills'].iloc[0], str):
                df['skills'] = df['skills'].apply(lambda x: [s.strip() for s in x.split(',')])
                
            profiles = df.to_dict('records')
        elif file.name.endswith('.json'):
            profiles = json.load(file)
        else:
            st.error("Unsupported file format. Please upload CSV or JSON.")
            return None
        
        # Validate profile structure
        for profile in profiles:
            if 'name' not in profile or 'skills' not in profile:
                st.warning(f"Some profiles are missing required fields (name, skills). Check your data format.")
                break
                
            # Ensure skills is a list
            if isinstance(profile.get('skills'), str):
                profile['skills'] = [s.strip() for s in profile['skills'].split(',')]
            
            # Ensure required fields exist
            profile['current_title'] = profile.get('current_title', '')
            profile['experience'] = profile.get('experience', '')
            profile['education'] = profile.get('education', '')
        
        return profiles
    except Exception as e:
        st.error(f"Error loading profiles: {str(e)}")
        return None

# Create sidebar for file uploads and settings
with st.sidebar:
    st.header("Upload Files")
    
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=['pdf'])
    profiles_file = st.file_uploader("Upload Candidate Profiles (CSV or JSON)", type=['csv', 'json'])
    
    st.header("Settings")
    top_n = st.slider("Number of top candidates to show", min_value=1, max_value=20, value=5)
    threshold = st.slider("Minimum match score (%)", min_value=0, max_value=100, value=50)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        use_gpt = st.checkbox("Use GPT for detailed analysis", value=True, 
                              help="Disable to use only keyword matching (faster but less accurate)")
        embedding_weight = st.slider("Embedding similarity weight", 0.0, 1.0, 0.4, 
                                    help="Higher values give more importance to semantic similarity")
        keyword_weight = st.slider("Keyword matching weight", 0.0, 1.0, 0.6,
                                  help="Higher values give more importance to keyword matching")
    
    process_btn = st.button("Process", type="primary")

# Main content area
if process_btn and jd_file and profiles_file:
    try:
        with st.spinner("Processing job description and profiles..."):
            # Extract job description text
            jd_text = extract_text_from_pdf(jd_file)
            
            if not jd_text:
                st.error("Could not extract text from the PDF. Please check the file.")
            else:
                # Display job description summary
                st.header("Job Description")
                with st.expander("View Job Description Text"):
                    st.write(jd_text)
                
                # Get job description embedding
                jd_embedding = get_embedding(jd_text)
                
                # Extract requirements from job description
                requirements = extract_requirements(jd_text)
                
                # Display extracted requirements
                st.subheader("Key Requirements")
                req_col1, req_col2 = st.columns(2)
                with req_col1:
                    st.write("**Required Skills:**")
                    st.write(", ".join(requirements['required_skills']))
                    st.write("**Qualifications:**")
                    st.write(", ".join(requirements['qualifications']))
                with req_col2:
                    st.write("**Preferred Skills:**")
                    st.write(", ".join(requirements['preferred_skills']))
                    st.write("**Experience:**")
                    st.write(", ".join(requirements['experience']))
                
                # Load and process profiles
                profiles = load_profiles(profiles_file)
                
                if not profiles:
                    st.error("Could not load profiles. Please check the file format.")
                else:
                    st.header(f"Analyzing {len(profiles)} Candidate Profiles")
                    progress_bar = st.progress(0)
                    
                    results = []
                    batch_size = 5  # Process in batches to show progress
                    
                    for i, profile in enumerate(profiles):
                        # Update progress
                        progress_bar.progress((i + 1) / len(profiles))
                        
                        # Create profile text for embedding
                        profile_text = f"{profile['current_title']} {' '.join(profile['skills'])} {profile['experience']} {profile['education']}"
                        
                        # Get profile embedding
                        profile_embedding = get_embedding(profile_text)
                        
                        # Calculate embedding similarity
                        similarity_score = cosine_similarity(jd_embedding, profile_embedding) * 100  # Convert to percentage
                        
                        # Calculate keyword match score
                        keyword_match = calculate_keyword_match(profile, requirements)
                        
                        # Initial match score based on weighted average of embedding and keyword
                        initial_match_score = (
                            embedding_weight * similarity_score + 
                            keyword_weight * keyword_match['keyword_score']
                        )
                        
                        # Detailed analysis using GPT (if enabled)
                        if use_gpt:
                            analysis = analyze_profile_match(jd_text, profile, requirements, keyword_match)
                        else:
                            # Use only keyword matching if GPT is disabled
                            analysis = {
                                'match_score': round(initial_match_score),
                                'strengths': [f"Strong match in {category}" for category, score in 
                                             sorted(keyword_match['category_scores'].items(), key=lambda x: x[1], reverse=True)[:3] 
                                             if score > 70],
                                'gaps': [f"Low match in {category}" for category, score in 
                                        sorted(keyword_match['category_scores'].items(), key=lambda x: x[1])[:3] 
                                        if score < 50],
                                'overall_assessment': f"Based on keyword matching, this candidate has a {keyword_match['keyword_score']:.1f}% match with the job requirements."
                            }
                        
                        # Add results
                        results.append({
                            'profile': profile,
                            'similarity_score': similarity_score,
                            'keyword_score': keyword_match['keyword_score'],
                            'category_scores': keyword_match['category_scores'],
                            'match_score': analysis['match_score'],
                            'strengths': analysis['strengths'],
                            'gaps': analysis['gaps'],
                            'overall_assessment': analysis['overall_assessment']
                        })
                        
                        # Update progress and let UI refresh
                        if i % batch_size == 0:
                            time.sleep(0.1)
                    
                    # Sort results by match score
                    results.sort(key=lambda x: x['match_score'], reverse=True)
                    
                    # Display top candidates
                    st.header("Top Matching Candidates")
                    
                    if not results:
                        st.info("No candidates matched the job requirements.")
                    else:
                        filtered_results = [r for r in results if r['match_score'] >= threshold]
                        
                        if not filtered_results:
                            st.warning(f"No candidates met the minimum threshold of {threshold}%.")
                        else:
                            top_results = filtered_results[:top_n]
                            
                            # Display top candidates
                            for rank, result in enumerate(top_results, 1):
                                with st.container():
                                    st.subheader(f"{rank}. {result['profile']['name']} - {result['match_score']}% Match")
                                    
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        st.write(f"**Current Title:** {result['profile']['current_title']}")
                                        st.write(f"**Top Skills:** {', '.join(result['profile']['skills'][:5])}")
                                        st.write(f"**Embedding Similarity:** {result['similarity_score']:.1f}%")
                                        st.write(f"**Keyword Match:** {result['keyword_score']:.1f}%")
                                        
                                    with col2:
                                        st.write("**Overall Assessment:**")
                                        st.write(result['overall_assessment'])
                                    
                                    # Display strengths and gaps in columns
                                    strength_col, gap_col = st.columns(2)
                                    
                                    with strength_col:
                                        st.write("**Strengths:**")
                                        for strength in result['strengths']:
                                            st.write(f"- {strength}")
                                    
                                    with gap_col:
                                        st.write("**Gaps:**")
                                        for gap in result['gaps']:
                                            st.write(f"- {gap}")
                                    
                                    # Optional: Show category scores in an expander
                                    with st.expander("Detailed Category Scores"):
                                        for category, score in sorted(result['category_scores'].items(), key=lambda x: x[1], reverse=True):
                                            st.write(f"**{category.replace('_', ' ').title()}:** {score:.1f}%")
                                    
                                    st.divider()
                            
                            # Download results as CSV
                            results_df = pd.DataFrame([{
                                'Rank': i+1,
                                'Name': r['profile']['name'],
                                'Current Title': r['profile']['current_title'],
                                'Match Score': r['match_score'],
                                'Embedding Similarity': r['similarity_score'],
                                'Keyword Match': r['keyword_score'],
                                'Overall Assessment': r['overall_assessment'],
                                'Strengths': ', '.join(r['strengths']),
                                'Gaps': ', '.join(r['gaps'])
                            } for i, r in enumerate(filtered_results)])
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results CSV",
                                data=csv,
                                file_name="candidate_matches.csv",
                                mime="text/csv",
                            )
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

# Instructions when no files are uploaded
elif not jd_file or not profiles_file:
    st.info("Please upload a job description PDF and a candidate profiles file (CSV or JSON) to get started.")
    
    # Sample data format description
    with st.expander("Expected Data Format"):
        st.subheader("Candidate Profiles Format (CSV or JSON)")
        
        st.code("""
# CSV Example:
name,current_title,skills,experience,education
"John Doe","Senior Software Engineer","Python,Java,AWS,Docker,Machine Learning","7 years in software development","MS Computer Science, Stanford University"
"Jane Smith","Data Scientist","Python,R,SQL,TensorFlow,Statistics","5 years in data science","PhD in Statistics, MIT"

# JSON Example:
[
  {
    "name": "John Doe",
    "current_title": "Senior Software Engineer",
    "skills": ["Python", "Java", "AWS", "Docker", "Machine Learning"],
    "experience": "7 years in software development",
    "education": "MS Computer Science, Stanford University"
  },
  {
    "name": "Jane Smith",
    "current_title": "Data Scientist",
    "skills": ["Python", "R", "SQL", "TensorFlow", "Statistics"],
    "experience": "5 years in data science",
    "education": "PhD in Statistics, MIT"
  }
]
""")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit, OpenAI GPT-4, and OpenAI Embeddings")