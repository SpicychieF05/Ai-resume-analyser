import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure page
st.set_page_config(
    page_title="Smart Resume Analyser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional design with responsive optimization
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    /* General Body and Font */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        background-color: #f0f4f8;
    }

    /* Main container with responsive padding */
    .main {
        background-color: #f0f4f8;
        padding: 0.5rem 1rem 1rem 1rem;
    }

    /* Card styling with responsive design */
    .card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: box-shadow 0.3s ease-in-out;
        width: 100%;
        box-sizing: border-box;
    }
    .card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    /* Responsive Header */
    .title, h1.title {
        font-size: clamp(2.5rem, 6vw, 4rem);
        font-weight: 800;
        color: #0d6efd;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 1rem;
        line-height: 1.1;
        text-shadow: 0 2px 4px rgba(13, 110, 253, 0.1);
    }
    
    /* Social Media Buttons */
    .social-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .social-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        text-decoration: none;
        color: white;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        color: white;
        text-decoration: none;
    }
    
    .github-btn {
        background: linear-gradient(135deg, #333, #24292e);
    }
    
    .linkedin-btn {
        background: linear-gradient(135deg, #333, #24292e);
    }
    
    .twitter-btn {
        background: linear-gradient(135deg, #333, #24292e);
    }
    
    .subtitle, p.subtitle {
        font-size: clamp(1rem, 2.8vw, 1.3rem);
        color: #5a6268;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.4;
        padding: 0 1rem;
        font-weight: 500;
    }

    /* Score display with responsive sizing */
    .score-display {
        text-align: center;
        padding: 1rem 0;
    }
    
    .score-display .metric-container {
        margin: 0 auto;
        max-width: 300px;
    }

    /* Responsive text areas and file uploads */
    .stTextArea textarea {
        font-size: clamp(0.8rem, 2vw, 1rem) !important;
        line-height: 1.4 !important;
    }
    
    .stFileUploader {
        margin-bottom: 1rem;
    }

    /* Video container responsive */
    .stVideo {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    iframe {
        max-width: 100% !important;
        height: auto !important;
        aspect-ratio: 16/9;
    }

    /* Progress bar styling */
    .stProgress {
        margin: 1rem 0;
    }

    /* Responsive columns for mobile */
    div[data-testid="column"] {
        padding: 0 0.5rem;
    }

    /* Footer with responsive design */
    .footer {
        text-align: center;
        padding: 1.5rem 1rem;
        margin-top: 2rem;
        color: #718096;
        font-size: clamp(0.8rem, 2vw, 0.9rem);
        line-height: 1.4;
    }
    
    .footer p {
        margin-bottom: 0.5rem;
    }
    
    .footer a {
        word-break: break-word;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Mobile-specific optimizations */
    @media (max-width: 768px) {
        .main {
            padding: 0.25rem 0.5rem 0.5rem 0.5rem;
        }
        
        .card {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 8px;
        }
        
        .title, h1.title {
            margin-top: -0.5rem;
            margin-bottom: 0.75rem;
        }
        
        .social-buttons {
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        
        .social-btn {
            width: 40px;
            height: 40px;
            font-size: 1.1rem;
        }
        
        .subtitle, p.subtitle {
            margin-bottom: 1.5rem;
            padding: 0 0.5rem;
        }
        
        div[data-testid="column"] {
            padding: 0 0.25rem;
            margin-bottom: 1rem;
        }
        
        .stTextArea textarea {
            min-height: 150px !important;
        }
        
        .footer {
            padding: 1rem 0.5rem;
            margin-top: 1.5rem;
        }
        
        /* Stack columns on mobile */
        div[data-testid="column"]:first-child {
            margin-bottom: 1rem;
        }
    }

    /* Extra small screens */
    @media (max-width: 480px) {
        .main {
            padding: 0.25rem;
        }
        
        .card {
            padding: 0.75rem;
            border-radius: 6px;
        }
        
        .stTextArea textarea {
            min-height: 120px !important;
            font-size: 0.85rem !important;
        }
        
        .social-buttons {
            gap: 0.5rem;
        }
        
        .social-btn {
            width: 38px;
            height: 38px;
            font-size: 1rem;
        }
        
        .subtitle, p.subtitle {
            padding: 0 0.25rem;
        }
    }

    /* Large screens optimization */
    @media (min-width: 1200px) {
        .main {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .card {
            padding: 2.5rem;
            margin-bottom: 2rem;
        }
        
        .footer {
            padding: 2rem;
            margin-top: 3rem;
        }
    }

    /* Ensure proper spacing for metrics */
    div[data-testid="metric-container"] {
        background-color: transparent !important;
        border: none !important;
        padding: 1rem !important;
    }
    
    /* Responsive subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        font-size: clamp(1.2rem, 3vw, 1.5rem) !important;
        margin-bottom: 1rem !important;
    }
    
    /* Responsive bullet points */
    .stMarkdown ul li {
        font-size: clamp(0.85rem, 2.2vw, 1rem) !important;
        line-height: 1.5 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Info message styling */
    .stInfo {
        margin: 1rem 0 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    /* Spinner responsiveness */
    .stSpinner {
        text-align: center !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from a PDF file


def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.

    Args:
        file (PdfReader): The PDF file to extract text from.

    Returns:
        str: The extracted text.
    """
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on their job description


def rank_resumes(job_description, resumes):
    """
    Ranks resumes based on their similarity to the job description.

    Args:
        job_description (str): The job description.
        resumes (list): A list of resume texts.

    Returns:
        list: A list of cosine similarities between the job description and each resume.
    """
    # Combine job description with resumes
    documents = [job_description] + resumes

    # Create a TfidfVectorizer and fit it to the documents
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents).toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(
        [job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Function to generate key points based on resume content


def generate_key_points(resume_text):
    """
    Generates key points based on the content of the resume.

    Args:
        resume_text (str): The text of the resume.

    Returns:
        list: A list of key points for resume improvement.
    """
    key_points = set()
    if "web developer" in resume_text.lower():
        key_points.add(
            "Add Some Web Development Course Certificates to your Resume.")

    if "software engineer" in resume_text.lower():
        key_points.add(
            "Highlight your coding projects and include links to GitHub repositories.")

    if "data scientist" in resume_text.lower():
        key_points.add(
            "Include Data Science Certifications and Kaggle competition experience.")

    if "database administrator" in resume_text.lower():
        key_points.add(
            "Showcase SQL expertise and any database management certifications.")

    if "systems administrator" in resume_text.lower():
        key_points.add(
            "Highlight experience with system configurations, backups, and troubleshooting.")

    if "network engineer" in resume_text.lower():
        key_points.add(
            "List networking certifications like CCNA, CCNP to boost your profile.")

    if "ux/ui designer" in resume_text.lower():
        key_points.add(
            "Add a portfolio link showcasing UI/UX design projects.")

    if "it security analyst" in resume_text.lower():
        key_points.add(
            "Include cybersecurity certifications like CEH, CISSP, or CompTIA Security+.")

    if "cloud engineer" in resume_text.lower():
        key_points.add(
            "Mention cloud platform expertise (AWS, Azure, GCP) and relevant certifications.")

    if "machine learning engineer" in resume_text.lower():
        key_points.add(
            "Showcase machine learning projects and model deployments.")

    if "devops engineer" in resume_text.lower():
        key_points.add(
            "Highlight experience with CI/CD, Docker, Kubernetes, and automation tools.")

    if "business analyst" in resume_text.lower():
        key_points.add(
            "Include experience with data visualization tools like Power BI or Tableau.")

    if "full stack developer" in resume_text.lower():
        key_points.add(
            "Mention proficiency in both frontend and backend technologies.")

    if "cybersecurity specialist" in resume_text.lower():
        key_points.add(
            "List certifications and experience in ethical hacking, threat analysis, and risk management.")

    if "ai engineer" in resume_text.lower():
        key_points.add(
            "Mention deep learning, NLP experience, and AI framework expertise.")

    if "game developer" in resume_text.lower():
        key_points.add(
            "Showcase projects using Unity, Unreal Engine, or game development tools.")

    if "technical support specialist" in resume_text.lower():
        key_points.add(
            "Highlight problem-solving skills and IT support experience.")

    if "blockchain developer" in resume_text.lower():
        key_points.add(
            "Showcase blockchain project experience and knowledge of smart contracts.")

    if "embedded systems engineer" in resume_text.lower():
        key_points.add(
            "Mention experience with microcontrollers, IoT devices, and real-time systems.")

    if "robotics engineer" in resume_text.lower():
        key_points.add(
            "Highlight robotics programming skills and hands-on experience.")

    if "quantitative analyst" in resume_text.lower():
        key_points.add(
            "Showcase experience in quantitative finance, modeling, and risk analysis.")

    if "android developer" in resume_text.lower():
        key_points.add("Add Some Android Course Certificates to your Resume.")

    if "leadership" in resume_text.lower():
        key_points.add("Highlight your leadership experiences.")

    if "project management" in resume_text.lower():
        key_points.add("Include specific projects you've managed.")

    if "certification" in resume_text.lower():
        key_points.add("List relevant certifications.")

    if "skills" in resume_text.lower():
        key_points.add("Emphasize your technical skills.")

    if "teamwork" in resume_text.lower():
        key_points.add("Showcase your teamwork abilities.")

    if "problem-solving" in resume_text.lower():
        key_points.add("Detail your problem-solving skills.")

    if "awards" in resume_text.lower():
        key_points.add("Mention any awards or recognitions received.")

    if "volunteer" in resume_text.lower():
        key_points.add("Include volunteer experiences.")

    if "communication" in resume_text.lower():
        key_points.add("Highlight your communication skills.")

    if "adaptability" in resume_text.lower():
        key_points.add("Discuss your adaptability in changing environments.")

    if "software" in resume_text.lower():
        key_points.add("Mention your proficiency in specific software.")

    if "metrics" in resume_text.lower():
        key_points.add("Include metrics to quantify your achievements.")

    if "pressure" in resume_text.lower():
        key_points.add("Highlight your ability to work under pressure.")

    if "cross-functional" in resume_text.lower():
        key_points.add("Discuss your experience with cross-functional teams.")

    if "strategic planning" in resume_text.lower():
        key_points.add("Mention your strategic planning skills.")

    if "budgeting" in resume_text.lower():
        key_points.add(
            "Include any experience with budgeting or financial management.")

    if "customer service" in resume_text.lower():
        key_points.add("Highlight your customer service skills.")

    if "data analysis" in resume_text.lower():
        key_points.add("Discuss your experience with data analysis.")

    if "mentoring" in resume_text.lower():
        key_points.add("Mention your ability to mentor or train others.")

    if "conflict resolution" in resume_text.lower():
        key_points.add("Include your experience with conflict resolution.")

    if "creativity" in resume_text.lower():
        key_points.add("Highlight your creativity and innovation.")

    if "technical writing" in resume_text.lower():
        key_points.add("Highlight your skills in technical writing.")

    return list(key_points)


# Header
st.markdown('<h1 class="title">Smart Resume Analyser</h1>',
            unsafe_allow_html=True)

# Social Media Buttons
st.markdown("""
<div class="social-buttons">
    <a href="https://github.com/SpicychieF05" target="_blank" class="social-btn github-btn" title="GitHub">
        <i class="fab fa-github"></i>
    </a>
    <a href="https://linkedin.com/in/chirantan-mallick" target="_blank" class="social-btn linkedin-btn" title="LinkedIn">
        <i class="fab fa-linkedin-in"></i>
    </a>
    <a href="https://twitter.com/chirantan_mallick" target="_blank" class="social-btn twitter-btn" title="Twitter/X">
        <i class="fab fa-twitter"></i>
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="subtitle">Upload your resume and a job description to see the match score.</p>',
            unsafe_allow_html=True)

# --- Inputs Card ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    job_description = st.text_area(
        "Job Description",
        height=250,
        placeholder="Paste job description here...",
        help="Enter the job description you want to match against"
    )

with col2:
    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=['pdf'],
        help="Upload your PDF resume file"
    )

    # Add some spacing for mobile
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

if uploaded_file and job_description:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        scores = rank_resumes(job_description, [resume_text])
        score_percentage = round(scores[0] * 100, 2)
        key_points = generate_key_points(resume_text)

    # --- Results Section ---
    # Score display with better mobile layout
    st.markdown('<div class="score-display">', unsafe_allow_html=True)

    # Create responsive metric display
    score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
    with score_col2:
        st.metric("Match Score", f"{score_percentage}%")
        st.progress(int(score_percentage))

    st.markdown('</div>')

    st.divider()

    # Key Points with better mobile formatting
    st.subheader("📝 Resume Suggestions")
    if key_points:
        # Create a more readable list for mobile
        suggestions_container = st.container()
        with suggestions_container:
            for i, point in enumerate(key_points, 1):
                st.markdown(f"**{i}.** {point}")
    else:
        st.info("No specific suggestions at the moment. Your resume looks good!")

    # --- Video Recommendations Section ---
    st.subheader("🎥 Recommended Videos")

    video_recommendations = {
        "Add Some Web Development Course Certificates to your Resume.": "https://www.youtube.com/watch?v=DPnqb74Smug",
        "Highlight your coding projects and include links to GitHub repositories.": "https://www.youtube.com/watch?v=wpISo9TNjfU",
        "Include Data Science Certifications and Kaggle competition experience.": "https://www.youtube.com/watch?v=ua-CiDNNj30",
        "Showcase SQL expertise and any database management certifications.": "https://www.youtube.com/watch?v=7S_tz1z_5bA",
        "Highlight experience with system configurations, backups, and troubleshooting.": "https://www.youtube.com/watch?v=Y8hXt9Pyv4M",
        "List networking certifications like CCNA, CCNP to boost your profile.": "https://www.youtube.com/watch?v=qiQR5rTSshw",
        "Add a portfolio link showcasing UI/UX design projects.": "https://www.youtube.com/watch?v=GJjMjSDZhtU",
        "Include cybersecurity certifications like CEH, CISSP, or CompTIA Security+.": "https://www.youtube.com/watch?v=vv6tqxLjjRI",
        "Mention cloud platform expertise (AWS, Azure, GCP) and relevant certifications.": "https://www.youtube.com/watch?v=ulprqHHWlng",
        "Showcase machine learning projects and model deployments.": "https://www.youtube.com/watch?v=Gv9_4yMHFhI",
        "Highlight experience with CI/CD, Docker, Kubernetes, and automation tools.": "https://www.youtube.com/watch?v=WMy1nFofmZ8",
        "Include experience with data visualization tools like Power BI or Tableau.": "https://www.youtube.com/watch?v=AGrl-H87pRU",
        "Mention proficiency in both frontend and backend technologies.": "https://www.youtube.com/watch?v=nu_pCVPKzTk",
        "List certifications and experience in ethical hacking, threat analysis, and risk management.": "https://www.youtube.com/watch?v=3Kq1MIfTWCE",
        "Mention deep learning, NLP experience, and AI framework expertise.": "https://www.youtube.com/watch?v=aircAruvnKk",
        "Showcase projects using Unity, Unreal Engine, or game development tools.": "https://www.youtube.com/watch?v=IxiWurU0-tE",
        "Highlight problem-solving skills and IT support experience.": "https://www.youtube.com/watch?v=Fi1w8rJjMq8",
        "Showcase blockchain project experience and knowledge of smart contracts.": "https://www.youtube.com/watch?v=HXoVSbwWUIk",
        "Mention experience with microcontrollers, IoT devices, and real-time systems.": "https://www.youtube.com/watch?v=l1M3Krt-fQY",
        "Highlight robotics programming skills and hands-on experience.": "https://www.youtube.com/watch?v=xS3s7I6g1Zw",
        "Showcase experience in quantitative finance, modeling, and risk analysis.": "https://www.youtube.com/watch?v=smbvZ6wBXlA",
        "Add Some Android Course Certificates to your Resume.": "https://youtu.be/HyU4vkZ2NB8",
        "Highlight your leadership experiences.": "https://www.youtube.com/watch?v=JRlKKljzGI0",
        "Include specific projects you've managed.": "https://www.youtube.com/watch?v=JrcjOCq7pLg",
        "List relevant certifications.": "https://www.youtube.com/watch?v=m7103Kj6sSc",
        "Emphasize your technical skills.": "https://www.youtube.com/watch?v=366t4jL0b54",
        "Showcase your teamwork abilities.": "https://www.youtube.com/watch?v=wXWqYf4c97E",
        "Detail your problem-solving skills.": "https://youtu.be/hiqoCvPs_Jc?si=trgnc3ek8mQQVLZU",
        "Mention any awards or recognitions received.": "https://www.youtube.com/watch?v=sO58z-V8V5Q",
        "Include volunteer experiences.": "https://www.youtube.com/watch?v=j9uN9_4mD94",
        "Highlight your communication skills.": "https://www.youtube.com/watch?v=icudf_w_pqU&ab_channel=ApnaCollege",
        "Discuss your adaptability in changing environments.": "https://www.youtube.com/watch?v=swbK6rppnMo",
        "Mention your proficiency in specific software.": "https://www.youtube.com/watch?v=I8E6uR63uXI",
        "Include metrics to quantify your achievements.": "https://www.youtube.com/watch?v=UYJ1gk4g2kY",
        "Highlight your ability to work under pressure.": "https://www.youtube.com/watch?v=WTxSmnZupxM",
        "Discuss your experience with cross-functional teams.": "https://www.youtube.com/watch?v=z75CBV7KA60",
        "Mention your strategic planning skills.": "https://www.youtube.com/watch?v=gnnPNzGhO4E",
        "Include any experience with budgeting or financial management.": "https://www.youtube.com/watch?v=geHpLxMBuVk",
        "Highlight your customer service skills.": "https://www.youtube.com/watch?v=yyRW6SJW1SI",
        "Discuss your experience with data analysis.": "https://www.youtube.com/watch?v=VPLXx0fou4M",
        "Mention your ability to mentor or train others.": "https://www.youtube.com/watch?v=1Skym90ZxWo",
        "Include your experience with conflict resolution.": "https://www.youtube.com/watch?v=aDzw37zefjA",
        "Highlight your creativity and innovation.": "https://www.youtube.com/watch?v=cL_8gRkKV8Q",
        "Highlight your skills in technical writing.": "https://www.youtube.com/watch?v=ueP7IGumrl0",
    }

    recommended_videos_found = False
    video_count = 0
    max_videos_mobile = 3  # Limit videos on mobile for better performance

    for point in key_points:
        if point in video_recommendations:
            if video_count < max_videos_mobile:  # Show first 3 videos to avoid overwhelming mobile users
                st.markdown(f"**Related to:** {point}")
                st.video(video_recommendations[point])
                st.markdown("---")
                recommended_videos_found = True
                video_count += 1
            else:
                break

    if not recommended_videos_found:
        st.info("🎯 No specific video recommendations for the given resume. Upload a resume to get personalized suggestions!")
    elif video_count >= max_videos_mobile and len([p for p in key_points if p in video_recommendations]) > max_videos_mobile:
        st.info(
            f"📱 Showing top {max_videos_mobile} video recommendations for better mobile experience.")

else:
    # Enhanced info message with better mobile formatting
    st.info("📋 **Getting Started**\n\n1. **Upload your resume** (PDF format only)\n2. **Paste the job description** you want to match against\n3. **Get instant analysis** with match score and improvement suggestions\n4. **Watch recommended videos** to enhance your resume")

    # Add some helpful tips
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        - Ensure your resume is in PDF format
        - Use clear, readable fonts in your resume
        - Include relevant keywords from the job description
        - Make sure the job description is complete and detailed
        - The more specific the job description, the better the analysis
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Smart Resume Analyser | All Rights Reserved</p>
    <p style="margin-top: 1rem;">
        Developed by <a href="https://linktr.ee/chirantan_mallick" target="_blank" style="color: #0d6efd; text-decoration: none;">Chirantan Mallick</a>
    </p>
</div>
""", unsafe_allow_html=True)
