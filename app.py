# app.py - Modern Job Search Application with Tabbed Interface
import os
import sys
import json
import gradio as gr
import threading
import pandas as pd
from typing import Any, Dict, Tuple, List
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from dotenv import load_dotenv
load_dotenv()

from agents.job_lookup_agent import search_jobs, advanced_job_search

from agents.resume_matcher_agent import ResumeMatcher
from utils.llm_client import LLMClient

# Constants
CURRENT_UTC_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_USER = "Admin"

def validate_api_keys(serp_api_key: str = None, nebius_api_key: str = None) -> Tuple[bool, str]:
    """Validate provided API keys"""
    if not serp_api_key:
        return False, "SerpAPI key is required for job searching"
    if not nebius_api_key:
        return False, "Nebius API key is required for advanced search"
    return True, "API keys validated"

def process_search_with_timeout(
    query: str,
    include_salary: bool = True,
    location: str = "Canada",
    level: str = "Senior",
    remote: bool = False,  
    timeout: int = 100,
    use_llm: bool = True,
    serp_api_key: str = None,
    nebius_api_key: str = None
) -> Tuple[str, dict]:
    """Process job search with timeout and API key handling, returning raw + table formats."""
    
    if not serp_api_key:
        return "Please provide your SerpAPI key", {"raw": [], "table": []}
    if use_llm and not nebius_api_key:
        return "Please provide your Nebius API key for advanced search", {"raw": [], "table": []}
    if not query or not query.strip():
        return "Please enter a search query", {"raw": [], "table": []}
    
    result_container = {"status": "Processing...", "data": {"raw": [], "table": []}}

    def search_worker():
        try:
            search_query = query.strip()
            if use_llm:
                search_result = advanced_job_search(
                    query=search_query,
                    location=location,
                    remote=remote,
                    level=level,
                    use_llm=True,
                    serp_api_key=serp_api_key,
                    nebius_api_key=nebius_api_key
                )
                if not search_result["success"]:
                    result_container["status"] = f"Search failed: {search_result.get('error', 'Unknown error')}"
                    return
                jobs_data = search_result["jobs"]
            else:
                raw_results = search_jobs(
                    query=search_query,
                    location=location,
                    remote=remote,
                    level=level,
                    serp_api_key=serp_api_key
                )
                try:
                    jobs_data = json.loads(raw_results)
                except json.JSONDecodeError:
                    result_container["status"] = "Error parsing results"
                    return

            if not jobs_data or not isinstance(jobs_data, list):
                result_container["status"] = "No jobs found"
                return

            table_data = []
            for job in jobs_data:
                title = job.get("title", "N/A")
                company = job.get("company_name", "N/A")
                job_location = job.get("location", "N/A")
                salary = job.get("salary", "N/A")
                is_remote = job.get("remote", "No")
                posted_date = job.get("posted_at", "N/A")
                apply_link = job.get("link", "#")

                # Clean apply link
                if apply_link and '<a href="' in apply_link:
                    apply_link = apply_link.replace('<a href="', '').replace('" target="_blank">Apply</a>', '').replace('"', '')
                formatted_link = (
                    f'<a href="{apply_link}" target="_blank" style="color: #3b82f6; text-decoration: none; font-weight: 500;">Apply →</a>'
                    if apply_link not in ["N/A", "#"]
                    else "N/A"
                )

                location_display = job_location
                if location_display.lower() in ["anywhere", "remote"]:
                    location_display = "🌍 Remote Worldwide"
                elif "remote" in location_display.lower():
                    location_display = f"🏠 {location_display}"

                remote_status = "Yes" if str(is_remote).lower() in ["yes", "true", "remote", "1"] or "remote" in job_location.lower() else "No"

                row = [
                    title,
                    company,
                    location_display,
                    salary if include_salary else "",
                    remote_status,
                    posted_date,
                    formatted_link
                ]

                if not include_salary:
                    row.pop(3)  # Remove salary column

                table_data.append(row)

            result_container["status"] = f"Found {len(table_data)} jobs using {'advanced search' if use_llm else 'basic search'}"
            result_container["data"] = {
                "raw": jobs_data,
                "table": table_data
            }

        except Exception as e:
            result_container["status"] = f"Search failed: {str(e)}"
            result_container["data"] = {"raw": [], "table": []}

    # Run search in thread with timeout
    search_thread = threading.Thread(target=search_worker)
    search_thread.daemon = True
    search_thread.start()
    search_thread.join(timeout)

    if search_thread.is_alive():
        return "Search timed out. Please try again with a more specific query.", {"raw": [], "table": []}

    return result_container["status"], result_container["data"]

def normalize_data(data, include_salary=True):
    import pandas as pd
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    field_map = {
        "title": "Job Title",
        "company_name": "Company",
        "location": "Location",
        "remote": "Remote",
        "posted_at": "Posted",
        "link": "Apply Link"
    }

    if include_salary:
        field_map["salary"] = "Salary"

    # Rename only if the column exists
    cols_to_rename = {k: v for k, v in field_map.items() if k in df.columns}
    df = df.rename(columns=cols_to_rename)
    df = df[list(cols_to_rename.values())]
    return df

def export_csv(dataframe, include_salary=True):
    if not dataframe:
        return gr.update(visible=False)

    try:
        df = normalize_data(dataframe, include_salary)

        # Clean HTML from Apply Link column if present
        if 'Apply Link' in df.columns:
            df['Apply Link'] = (
                df['Apply Link']
                .astype(str)
                .str.replace(r'<.*?>', '', regex=True)
                .str.replace('Apply →', '')
                .str.strip()
            )

        filename = f"job_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')

        return gr.update(value=filename, visible=True)
    except Exception as e:
        print(f"CSV export error: {e}")
        return gr.update(visible=False)

def export_json(dataframe, include_salary=True):
    """Export DataFrame to JSON"""
    if not dataframe:
        return gr.update(visible=False)
    
    try:
        df = normalize_data(dataframe, include_salary)
        # Clean Apply Link column for JSON (remove HTML)
        if 'Apply Link' in df.columns:
            df['Apply Link'] = df['Apply Link'].astype(str).str.replace(r'<.*?>', '', regex=True).str.replace('Apply →', '').str.strip()
        
        # Generate filename with timestamp
        filename = f"job_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        df.to_json(filename, orient='records', indent=2)
        
        return gr.update(value=filename, visible=True)
    except Exception as e:
        print(f"JSON export error: {e}")
        return gr.update(visible=False)

    
def analyze_resume_and_match(
    resume_file,
    job_text: str,
    nebius_key: str,
    progress=gr.Progress()
) -> Tuple[float, float, List[List[str]], str, str]:
    try:
        if not nebius_key:
            return (
                0,
                0,
                [],
                "Error: Please provide your Nebius API key",
                "Please configure your API key first"
            )
        
        if not resume_file or not job_text.strip():
            return (
                0,
                0,
                [],
                "Error: Please provide both resume file and job description",
                "Upload your resume and paste the job description text"
            )
        
        try:
            progress(0.2, desc="Reading resume...")
            print(f"Resume file type: {type(resume_file)}")
            print(f"Resume file name: {resume_file.name}")
            
            # Read the PDF file using PyMuPDF (fitz)
            try:
                import fitz
                doc = fitz.open(resume_file.name)
                resume_text = ""
                for page in doc:
                    resume_text += page.get_text()
                doc.close()
            except Exception as pdf_error:
                print(f"Error reading PDF: {pdf_error}")
                return (
                    0,
                    0,
                    [],
                    f"Error reading PDF: {str(pdf_error)}",
                    "Please ensure your resume is a valid PDF file"
                )
            
            progress(0.2, desc="Initializing matcher...")
            # Initialize LLM client correctly with API key
            try:
                llm_client = LLMClient(api_key=nebius_key)  
                print("LLM client initialized successfully")
            except Exception as llm_error:
                print(f"Error initializing LLM client: {llm_error}")
                return (
                    0,
                    0,
                    [],
                    f"Error initializing AI client: {str(llm_error)}",
                    "Please check your API key and try again"
                )
            
            matcher = ResumeMatcher(
                llm_client=llm_client,
                current_user=CURRENT_USER,
                current_time=CURRENT_UTC_TIME
            )
            
            progress(0.4, desc="Analyzing resume...")
            resume_data = matcher.analyze_resume(resume_text)
            
            progress(0.8, desc="Preparing job data...")
            job_data = matcher.parse_job_from_text(job_text)
            
            progress(0.9, desc="Calculating match...")
            result = matcher.calculate_match(resume_data, job_data)
            
            # Convert skills analysis to table format
            skills_table = [
                [
                    skill["skill"],
                    skill["status"],
                    skill["found_in_resume"],
                    skill["relevance_score"]
                ]
                for skill in result.get("skills_analysis", [])
            ]
            
            progress(1.0, desc="Done!")
            return (
                float(result.get("match_score", 0)),
                float(result.get("confidence_score", 0)),
                skills_table,
                result.get("detailed_analysis", "No detailed analysis available"),
                "\n".join(result.get("improvement_suggestions", ["No suggestions available"])),
                resume_data,  
                job_data
            )
            
        except Exception as e:
            print(f"Error in resume analysis: {e}")
            return (
                0,
                0,
                [],
                f"Error processing resume: {str(e)}",
                "Please try again with a different PDF file"
            )
            
    except Exception as e:
        print(f"Error in match analysis: {e}")
        return (
            0,
            0,
            [],
            f"Analysis failed: {str(e)}",
            "An error occurred during analysis"
        )
    
def generate_cover_letter_fn(resume_json, job_json, api_key):
    try:
        llm_client = LLMClient(api_key=api_key)
        matcher = ResumeMatcher(
            llm_client=llm_client,
            current_user=CURRENT_USER,
            current_time=CURRENT_UTC_TIME
        )
        return matcher.generate_cover_letter(resume_json, job_json)
    except Exception as e:
        return f"❌ Error generating cover letter: {str(e)}"

def create_interface():
    """Create modern Gradio interface with tabbed layout"""
    
    # Load external CSS
    with open("static/styles.css", "r") as f:
        css = f.read()
    
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    def clear_search_tab():
        """Clear all components in the Search & Results tab"""
        return [
            "",  # search_input
            "Advanced Search (AI-Enhanced)",  # search_method
            "Senior",  # exp_level
            "Canada",  # location
            True,  # show_salary
            False,  # remote_only
            "",  # status_display
            None,  # results_table
            gr.update(visible=False),  # export_section
            gr.update(value=None, visible=False),  # csv_file
            gr.update(value=None, visible=False),  # json_file
            None  # raw_data_state
        ]
    
    def clear_resume_matcher_tab():
        """Clear all components in the Resume Matcher tab"""
        return [
            None,  # resume_file
            "",   # job_textbox
            0,    # match_score
            0,    # confidence_score
            None, # skills_analysis
            "",   # analysis_details
            "",   # suggestions
            None, # resume_json_state
            None, # job_json_state
            gr.update(visible=False),  # generate_cover_btn
            gr.update(visible=False, value="")  # cover_letter_output
        ]

    with gr.Blocks(title="Career Compass AI", css=css, theme=theme) as interface:
        # Full width header
        with gr.Group(elem_classes=["header-section"]):
            with gr.Group(elem_classes=["header-content"]):
                    gr.Markdown("""
                    <div class='app-header'>
                        <h1>🧭 Career Compass AI</h1>
                        <p class='app-description'>Your all-in-one AI-powered career assistant for job search, resume optimization, and professional document generation.</p>
                        <div class='header-features'>
                            <span>🔍 Smart Job Search</span>
                            <span>📊 Resume Analysis</span>
                            <span>✍️ Cover Letter Generator</span>
                            <span>🎯 Skills Matcher</span>
                        </div>
                    </div>
                """)
                
        # Main Tabbed Interface
        with gr.Tabs(elem_classes=["fixed-width-container", "main-content"]) as tabs:
            
            # Tab 1: API Configuration
            with gr.TabItem("🔑 API Configuration", elem_classes=["tab-content"]):
                with gr.Group(elem_classes=["api-config-section"]):
                    gr.Markdown("### API Keys Setup")
                    gr.Markdown("""
                    This tool requires two API keys to function properly:
                    
                    **🔗 [SerpAPI](https://serpapi.com)** - For job searching (Required for all searches)
                    - Sign up for free account and get API key
                    - Used for accessing job search data from multiple job boards
                    
                    **🤖 [Nebius](https://nebius.ai)** - For AI-powered filtering (Required for Advanced Search)
                    - Advanced AI model for intelligent job parsing and filtering
                    - Provides better accuracy in matching requirements
                    
                    **🔒 Security Note:** Your API keys are only stored temporarily in memory during your session and are never saved to disk.

                                            
                    """)
                    
                    with gr.Row():
                        serp_api_key = gr.Textbox(
                            label="SerpAPI Key",
                            placeholder="Enter your SerpAPI key here...",
                            type="password",
                            value=os.environ.get("SERP_API_KEY", ""),
                            info="Required for all job searches",
                            elem_classes=["api-input"]
                        )
                        nebius_api_key = gr.Textbox(
                            label="Nebius API Key",
                            placeholder="Enter your Nebius API key here...",
                            type="password",
                            value=os.environ.get("NEBIUS_API_KEY", ""),
                            info="Required for AI-enhanced searches",
                            elem_classes=["api-input"]
                        )
                    
                    # API Status Display
                    api_status = gr.Markdown("⚠️ Please enter your API keys to start searching", elem_classes=["api-status"])
            
            # Tab 2: Search & Results
            with gr.TabItem("🔍 Search & Results", elem_classes=["tab-content"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Search Parameters Section
                        with gr.Group(elem_classes=["search-params-section"]):
                            gr.Markdown("### &nbsp;Search Parameters")
                            
                            search_input = gr.Textbox(
                                label="Job Title/Keywords",
                                placeholder="e.g., Python Developer, Full Stack Engineer, DevOps",
                                lines=2,
                                info="Enter job title, skills, or keywords",
                                elem_classes=["search-input"]
                            )
                            
                            # Method selection
                            with gr.Group(elem_classes=["method-radio"]):
                                gr.Markdown("**&nbsp; Search Method Selection**")
                                search_method = gr.Radio(
                                    choices=["Advanced Search (AI-Enhanced)", "Basic Search (Fast)"],
                                    value="Advanced Search (AI-Enhanced)",
                                    label="Choose Search Method",
                                    info="• Advanced: Uses AI for intelligent parsing and better filtering (30-60s)\n• Basic: Fast search with standard filtering (10-30s)",
                                    show_label=True
                                )
                            
                            with gr.Row():
                                exp_level = gr.Dropdown(
                                    choices=["Junior", "Mid-Level", "Senior", "Lead", "Principal"],
                                    value="Senior",
                                    label="Experience Level",
                                    info="Filter by experience level"
                                )
                                
                                location = gr.Dropdown(
                                    choices=[
                                        "Canada", 
                                        "United States", 
                                        "United Kingdom", 
                                        "Australia", 
                                        "Germany",
                                        "Netherlands",
                                        "Worldwide"
                                    ],
                                    value="Canada",
                                    label="Location",
                                    info="Preferred job location"
                                )
                            
                            with gr.Row():
                                show_salary = gr.Checkbox(
                                    label="Include Salary Info",
                                    value=True,
                                    info="Show salary information when available"
                                )
                                
                                remote_only = gr.Checkbox(
                                    label="Remote Positions Only",
                                    value=False,
                                    info="Filter ONLY for remote work opportunities"
                                )
                            
                            with gr.Row():
                                search_button = gr.Button(
                                "🔍 Search Jobs", 
                                variant="primary",
                                size="lg",
                                elem_classes=["search-button"]
                                )
                                clear_all_btn = gr.Button(
                                "🗑️ Clear All",
                                variant="secondary",
                                size="lg",
                                elem_classes=["clear-button"]
                                )
                            
                            # Status display
                            status_display = gr.Textbox(
                                label="Search Status",
                                interactive=False,
                                info="Current search status and results count",
                                elem_classes=["status-display"]
                            )
                
                # Quick Examples Section
                with gr.Group(elem_classes=["example-buttons"]):
                    gr.Markdown("### &nbsp; 🚀 Quick Examples")
                    gr.Markdown("*&nbsp; Click any example to populate the search form*")
                    
                    with gr.Row():
                        example_btn1 = gr.Button("🐍 Python Developer (Remote)", size="sm", variant="secondary", elem_classes=["example-button"])
                        example_btn2 = gr.Button("⚛️ Full Stack Engineer", size="sm", variant="secondary", elem_classes=["example-button"]) 
                        example_btn3 = gr.Button("🔧 DevOps Engineer (Senior)", size="sm", variant="secondary", elem_classes=["example-button"])
                        example_btn4 = gr.Button("⚡ React Developer (Entry)", size="sm", variant="secondary", elem_classes=["example-button"])
                
                # Results section
                with gr.Group(elem_classes=["results-section"]):
                    gr.Markdown("### &nbsp; Search Results")
                    
                    results_table = gr.DataFrame(
                        label="Job Listings",
                        wrap=True,
                        interactive=False,
                        elem_classes=["results-table"],
                        headers=["Job Title", "Company", "Location", "Salary", "Remote", "Posted", "Apply"],
                        datatype=["str", "str", "str", "str", "str", "str", "html"]
                    )

                    raw_data_state = gr.State() 
                    
                    # Export Section (Initially Hidden)
                    with gr.Group(elem_classes=["export-section"], visible=False) as export_section:
                        gr.Markdown("### &nbsp; Export Results")
                        with gr.Row():
                            export_csv_btn = gr.Button("📄 Export as CSV", elem_classes=["export-button"])
                            export_json_btn = gr.Button("📋 Export as JSON", elem_classes=["export-button", "json"])
                        
                        with gr.Row():
                            csv_file = gr.File(interactive=False, visible=False)
                            json_file = gr.File(interactive=False, visible=False)
            
            # Tab 3: Resume Matcher 
            with gr.TabItem("📄 Resume Matcher & Cover Letter Generation", elem_classes=["tab-content"]):
                resume_json_state = gr.State()
                job_json_state = gr.State()

                with gr.Group(elem_classes=["resume-matcher-section"]):
                    gr.Markdown("### &nbsp; 📄 Resume Analysis & Job Matching")
                    gr.Markdown(f"""&nbsp; Upload your resume and paste a job posting URL to get a detailed match analysis.
        
                Current User: {CURRENT_USER}
                Analysis Time (UTC): {CURRENT_UTC_TIME}""")
        
                    with gr.Row():
                        with gr.Column(scale=1):
                            resume_file = gr.File(
                                label="Upload Resume (PDF)",
                                file_types=[".pdf"],
                                elem_classes=["resume-upload"]
                            )
            
                        with gr.Column(scale=1):
                            job_textbox = gr.Textbox(
                                label="Paste Job Description Here",
                                placeholder="Paste full job description text...",
                                lines=15,
                                elem_classes=["manual-job-description"]
                            )
        
                    with gr.Row():
                        analyze_btn = gr.Button(
                        "🎯 Analyze Match",
                        variant="primary",
                        elem_classes=["analyze-button"]
                        )
                        clear_matcher_btn = gr.Button(
                        "🗑️ Clear All",
                        variant="secondary",
                        elem_classes=["clear-button"]
                        )

                    with gr.Group(elem_classes=["results-group"]):
                        with gr.Row():
                            match_score = gr.Number(
                                label="Match Score",
                                value=0,
                                minimum=0,
                                maximum=100,
                                interactive=False,
                                elem_classes=["score-display"]
                            )
                            confidence_score = gr.Number(
                                label="Confidence Score",
                                value=0,
                                minimum=0,
                                maximum=100,
                                interactive=False,
                                elem_classes=["score-display"]
                            )
            
                        skills_analysis = gr.DataFrame(
                            headers=["Required Skill", "Status", "Found in Resume", "Relevance Score"],
                            label="Skills Analysis",
                            interactive=False,
                            elem_classes=["results-table"]
                        )
            
                        with gr.Accordion("Detailed Analysis", open=False):
                            analysis_details = gr.Markdown(
                            elem_classes=["analysis-details"]
                        )
            
                        with gr.Accordion("Improvement Suggestions", open=False):
                            suggestions = gr.Markdown(
                            elem_classes=["improvement-suggestions"]
                        )
                        
                        # Cover Letter Section (Initially Hidden)
                        with gr.Group(elem_classes=["cover-letter-section"]) as cover_letter_section:
                            gr.Markdown("### &nbsp; Generate Cover Letter")
                            generate_cover_btn = gr.Button(
                                "✍️ Generate Cover Letter",
                                visible=False,
                                elem_classes=["cover-letter-button"]
                            )
                            cover_letter_output = gr.Textbox(
                                lines=20,
                                label="Generated Cover Letter",
                                interactive=False,
                                visible=False,
                                elem_classes=["cover-letter-output"]
                            )         

                    # Connect the analyze button
                    analyze_btn.click(
                        fn=analyze_resume_and_match,
                        inputs=[
                            resume_file,
                            job_textbox,
                            nebius_api_key
                        ],
                        outputs=[
                            match_score,
                            confidence_score,
                            skills_analysis,
                            analysis_details,
                            suggestions,
                            resume_json_state,  
                            job_json_state 
                        ],
                        show_progress=True
                    ).then(
                        # Show cover letter section when data is available
                        fn=lambda r, j: (gr.update(visible=True), gr.update(visible=True)) if r is not None and j is not None else (gr.update(visible=False), gr.update(visible=False)),
                        inputs=[resume_json_state, job_json_state],
                        outputs=[generate_cover_btn, cover_letter_output]
                    )

                    # generate cover letter button connection
                    generate_cover_btn.click(
                        fn=generate_cover_letter_fn,
                        inputs=[resume_json_state, job_json_state, nebius_api_key],
                        outputs=[cover_letter_output]
                    )

                    # Connect the clear button
                    clear_matcher_btn.click(
                        fn=clear_resume_matcher_tab,
                        outputs=[
                            resume_file,
                            job_textbox,
                            match_score,
                            confidence_score,
                            skills_analysis,
                            analysis_details,
                            suggestions,
                            resume_json_state,
                            job_json_state,
                            generate_cover_btn,
                            cover_letter_output
                        ]
                    )

            # Tab 4: Help & Documentation
            with gr.TabItem("📚 Help & Documentation", elem_classes=["tab-content"]):
                with gr.Group(elem_classes=["help-header"]):
                    gr.Markdown(f"""
                    # 📚 Application Documentation
        
                    &nbsp; **Latest Update:** {CURRENT_UTC_TIME}  
                    &nbsp; **By:** {CURRENT_USER} 
                    &nbsp; **Version:** 2.0.0
                    """)
    
                with gr.Tabs() as doc_tabs:
                    # Quick Start Guide
                    with gr.TabItem("🚀 Quick Start"):
                        with gr.Accordion("Getting Started", open=True):
                            gr.Markdown("""
                            ### 1. Configure API Keys
                            - Enter your SerpAPI and Nebius API keys in the API Configuration tab
                            - Keys are required for job searching and AI features
                
                            ### 2. Search for Jobs
                            - Use the Search & Results tab
                            - Enter job title or keywords
                            - Choose search method (Advanced or Basic)
                            - Set location and experience preferences
                
                            ### 3. Analyze Your Resume
                            - Use the Resume Matcher tab
                            - Upload your PDF resume
                            - Paste job description
                            - Get instant analysis and scores
                            """)

                    # Feature Details
                    with gr.TabItem("✨ Features"):
                        with gr.Accordion("Job Search", open=True):
                            gr.Markdown("""
                            ### 🤖 Advanced Search (AI-Enhanced)
                            - Uses LLM for intelligent parsing
                            - Higher precision matching
                            - 30-60 seconds processing
                
                            ### ⚡ Basic Search
                            - Direct API search
                            - 10-30 seconds processing
                            - Best for quick lookups
                            """)
            
                        with gr.Accordion("Resume Matcher"):
                            gr.Markdown("""
                            ### 🎯 Analysis Features
                            - Match & Confidence Scores
                            - Skills Analysis Table
                            - Detailed Breakdown
                            - Improvement Suggestions
                
                            ### ✍️ Cover Letter
                            - AI-Generated
                            - Context-Aware
                            - Customizable
                            """)

                    # Best Practices
                    with gr.TabItem("💡 Tips & Tricks"):
                        with gr.Accordion("Search Optimization", open=True):
                            gr.Markdown("""
                            ### Keywords
                            - Use specific skills (React, Python, AWS)
                            - Include job levels
                            - Combine role types
                
                            ### Location Strategy
                            - Remote Worldwide
                            - Specific Countries
                            - Hybrid Options
                            """)
            
                        with gr.Accordion("Resume Matcher Tips"):
                            gr.Markdown("""
                            - Upload clear PDF resumes
                            - Review skill analysis
                            - Use improvement suggestions
                            - Generate cover letter after good match
                            """)

                    # Security & Privacy
                    with gr.TabItem("🔒 Security"):
                        with gr.Accordion("Data Protection", open=True):
                            gr.Markdown("""
                            ### API Keys
                            - Memory-only storage
                            - HTTPS encryption
                            - Session-based
                
                            ### User Data
                            - No persistent storage
                            - Local processing
                            - No tracking
                            """)

                    # Troubleshooting
                    with gr.TabItem("🔧 Help"):
                        with gr.Accordion("Common Issues", open=True):
                            gr.Markdown("""
                            ### Search Problems
                            - No Results → Try broader terms
                            - Timeout → Use specific queries
                            - API Errors → Check keys
                
                            ### Resume Analysis
                            - PDF Errors → Check format
                            - Low Scores → Review suggestions
                            - Analysis Fails → Check input format
                            """)
            
                        with gr.Accordion("Support Links"):
                            gr.Markdown("""
                            - [SerpAPI Documentation](https://serpapi.com/search-api)
                            - [Nebius AI Platform](https://nebius.ai)
                            - [HuggingFace Space](https://huggingface.co/spaces/Agents-MCP-Hackathon/job-hunting-ai/tree/main)        
                            """)

                with gr.Group(elem_classes=["help-footer"]):
                    gr.Markdown("""
                    ---
                    *Need more help? Check our [documentation repository](https://huggingface.co/spaces/Agents-MCP-Hackathon/job-hunting-ai/blob/main/README.md) or [reach out to me](https://huggingface.co/mananshah296).*
                    """)
        
        # Search functionality
        def handle_search(query, method, salary, loc, level, remote, serp_key, nebius_key):
            if not serp_key:
                return "Please enter your SerpAPI key in the API Configuration tab", gr.DataFrame(value=[]), gr.Group.update(visible=False)
            if method == "Advanced Search (AI-Enhanced)" and not nebius_key:
                return "Please enter your Nebius API key for advanced search", gr.DataFrame(value=[]), gr.Group.update(visible=False)
            
            if not query or not query.strip():
                return "Please enter a search query", gr.DataFrame(value=[]), gr.Group.update(visible=False)
            
            # Determine which method to use
            use_advanced = method == "Advanced Search (AI-Enhanced)"
            
            # Show what we're searching for
            search_info = f"Searching for: '{query}' | Location: {loc} | Level: {level} | Remote Only: {'Yes' if remote else 'No'}"
            print(search_info)
            
            # Perform search with direct API key passing
            status, result = process_search_with_timeout(
                query=query,
                include_salary=salary,
                location=loc,
                level=level,
                remote=remote,
                timeout=60 if use_advanced else 30,
                use_llm=use_advanced,
                serp_api_key=serp_key,
                nebius_api_key=nebius_key
            )

            table_data = result.get("table", [])
            raw_data = result.get("raw", [])

            if salary:
                headers = ["Job Title", "Company", "Location", "Salary", "Remote", "Posted", "Apply"]
                column_types = ["str", "str", "str", "str", "str", "str", "html"]
            else:
                headers = ["Job Title", "Company", "Location", "Remote", "Posted", "Apply"]
                column_types = ["str", "str", "str", "str", "str", "html"]

            if table_data:
                return status, gr.DataFrame(
                    value=table_data,
                    headers=headers,
                    datatype=column_types
                ), gr.update(visible=True), raw_data
            else:
                return status, gr.DataFrame(
                    value=[],
                    headers=headers,
                    datatype=column_types
                ), gr.update(visible=False), []
            
        # Connect search button
        search_button.click(
            fn=handle_search,
            inputs=[
                search_input,
                search_method,
                show_salary,
                location,
                exp_level,
                remote_only,
                serp_api_key,
                nebius_api_key
            ],
            outputs=[status_display, results_table, export_section, raw_data_state],
            show_progress=True
        )

        # Connect the clear button
        clear_all_btn.click(
            fn=clear_search_tab,
                outputs=[
                    search_input,
                    search_method,
                    exp_level,
                    location,
                    show_salary,
                    remote_only,
                    status_display,
                    results_table,
                    export_section,
                    csv_file,
                    json_file,
                    raw_data_state
                ]
            )
        
        # Example button functions
        def set_example_1():
            return "Python Developer", "Advanced Search (AI-Enhanced)", True, "Canada", "Senior", True
        
        def set_example_2():
            return "Full Stack Engineer", "Basic Search (Fast)", True, "United States", "Mid-Level", False
        
        def set_example_3():
            return "DevOps Engineer", "Advanced Search (AI-Enhanced)", False, "Remote Worldwide", "Senior", True
        
        def set_example_4():
            return "React Developer", "Basic Search (Fast)", True, "United Kingdom", "Junior", False
        
        # Connect example buttons
        example_btn1.click(
            fn=set_example_1,
            outputs=[search_input, search_method, show_salary, location, exp_level, remote_only]
        )
        
        example_btn2.click(
            fn=set_example_2,
            outputs=[search_input, search_method, show_salary, location, exp_level, remote_only]
        )
        
        example_btn3.click(
            fn=set_example_3,
            outputs=[search_input, search_method, show_salary, location, exp_level, remote_only]
        )
        
        example_btn4.click(
            fn=set_example_4,
            outputs=[search_input, search_method, show_salary, location, exp_level, remote_only]
        )
        
        # Export functionality
        export_csv_btn.click(
            fn=lambda df, salary: export_csv(df, salary),
            inputs=[raw_data_state, show_salary],
            outputs=csv_file
        )
        
        export_json_btn.click(
            fn=lambda df, salary: export_json(df, salary),
            inputs=[raw_data_state, show_salary],
            outputs=json_file
        )
        
        # API key validation function
        def validate_keys(serp_key, nebius_key):
            if not serp_key and not nebius_key:
                return "⚠️ Please enter both API keys to get started"
            elif not serp_key:
                return "⚠️ SerpAPI key is required for all searches"
            elif not nebius_key:
                return "⚠️ Nebius API key is required for advanced search"
            else:
                return "✅ API keys configured and ready to search"
        
        # Connect key validation
        for key in [serp_api_key, nebius_api_key]:
            key.change(
                fn=validate_keys,
                inputs=[serp_api_key, nebius_api_key],
                outputs=api_status
            )
        # Trigger initial validation when interface loads
        interface.load(
            fn=validate_keys,
            inputs=[serp_api_key, nebius_api_key],
            outputs=api_status
        )
    return interface

# Main execution
if __name__ == "__main__":
    print("Starting Modern Job Search Application...")
    print(f"Current time: {CURRENT_UTC_TIME}")
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Create CSS file if it doesn't exist
    if not os.path.exists("static/styles.css"):
        print("Creating CSS file...")
        # You would need to create the CSS file separately or copy it from the previous artifact
        with open("static/styles.css", "w") as f:
            f.write("/* CSS file - please copy from the CSS artifact provided */")
    
    try:
        # Create and launch interface
        demo = create_interface()
        
        # Launch with correct parameters
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
        )
        
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        import traceback
        traceback.print_exc()