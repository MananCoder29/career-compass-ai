from typing import Dict, Any, List, Optional
# from langchain.tools import Tool
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.prompts import PromptTemplate
# from bs4 import BeautifulSoup
import requests
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)
import re  

def extract_json_from_text(text: str) -> str:
    """
    Extract JSON string from LLM response with markdown formatting.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    raise ValueError("No valid JSON found in LLM response")


class ResumeMatcher:
    def __init__(
        self,
        llm_client: Any,
        current_user: str,
        current_time: str
    ):
        self.llm = llm_client
        self.current_user = current_user
        self.current_time = current_time
        
        # self.tools = [
        #     Tool(
        #         name="job_scraper",
        #         func=self._scrape_job_posting,
        #         description="Scrapes job descriptions from URLs. Input should be a URL."
        #     )
        # ]
        
        logger.info("ResumeMatcher initialized successfully")

    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Helper method to get LLM response"""
        try:
            # Use the generate method directly as implemented in your LLMClient
            response = self.llm.generate(messages)
            
            # Handle the response based on your LLMClient's output format
            return response
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise

    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume and extract information"""
        if not resume_text or not resume_text.strip():
            raise ValueError("Empty response from LLM")

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert resume analyzer. Extract key information from the resume and return it in a structured JSON format."
                },
                {
                    "role": "user",
                    "content": f"""Analyze this resume and extract information:
{resume_text}

Return a JSON object with these exact keys:
{{
    "personal_info": {{
        "name": "Full name",
        "email": "Email address",
        "phone": "Phone number",
        "location": "Current location",
        "linkedin": "LinkedIn URL if available"
    }},
    "current_role": "Most recent/current job title",
    "years_of_experience": "Total years of professional experience as a number",
    "education": [
        {{
            "degree": "Degree name",
            "field": "Field of study",
            "institution": "Institution name",
            "year": "Year of completion"
        }}
    ],
    "technical_skills": ["List of technical skills"],
    "soft_skills": ["List of soft skills"],
    "industries": ["List of industries worked in"],
    "key_achievements": ["List of key achievements"],
    "certifications": ["List of relevant certifications"],
    "languages": ["List of languages known"]
}}"""
                }
            ]

            response_text = self._get_llm_response(messages)
            clean_json = extract_json_from_text(response_text)
            return json.loads(clean_json)
            
        except Exception as e:
            logger.error(f"Resume analysis error: {e}")
            return {
                "error": str(e),
                "timestamp": self.current_time
            }

#     def _scrape_job_posting(self, url: str) -> str:
#         """Scrape job posting from URL"""
#         try:
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#             }
#             response = requests.get(url, headers=headers)
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             for script in soup(["script", "style"]):
#                 script.decompose()
            
#             text = soup.get_text()
#             lines = (line.strip() for line in text.splitlines())
#             chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#             result = ' '.join(chunk for chunk in chunks if chunk)
#             print(result)
#             return result
            
#         except Exception as e:
#             logger.error(f"Error scraping job posting: {e}")
#             return f"Error: {str(e)}"

#     def parse_job(self, url: str) -> Dict[str, Any]:
#         """Parse job posting from URL"""
#         try:
#             job_text = self._scrape_job_posting(url)
            
#             messages = [
#                 {
#                     "role": "system",
#                     "content": "You are an expert at parsing job descriptions. Extract key information accurately."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"""Extract key information from this job description:
# {job_text}

# Return a JSON object with:
# {{
#     "title": "Job title",
#     "required_skills": ["List of required technical skills"],
#     "preferred_skills": ["List of preferred skills"],
#     "experience_required": "Years of experience required",
#     "education_required": "Required education level",
#     "responsibilities": ["List of key responsibilities"]
# }}"""
#                 }
#             ]

#             response_text = self._get_llm_response(messages)
#             return json.loads(response_text)
            
#         except Exception as e:
#             logger.error(f"Job parsing error: {e}")
#             return {
#                 "error": str(e),
#                 "timestamp": self.current_time
#             }
        
    def parse_job_from_text(self, job_text: str) -> Dict[str, Any]:
        """Parse manually pasted job description"""
        try:
            if not job_text.strip():
                raise ValueError("Empty job description text")

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at parsing job descriptions. Extract key information accurately."
                },
                {
                    "role": "user",
                    "content": f"""Extract key information from this job description:
                {job_text}

                Return a JSON object with:
                {{
                    "title": "Job title",
                    "required_skills": ["List of required technical skills"],
                    "preferred_skills": ["List of preferred skills"],
                    "experience_required": "Years of experience required",
                    "education_required": "Required education level",
                    "responsibilities": ["List of key responsibilities"]
                }}"""
            }
        ]

            response_text = self._get_llm_response(messages)
            clean_json = extract_json_from_text(response_text)
            return json.loads(clean_json)

        except Exception as e:
            logger.error(f"Job parsing error (manual input): {e}")
        return {
            "error": str(e),
            "timestamp": self.current_time
        }

    def calculate_match(
        self,
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate match between resume and job"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at matching resumes to job requirements. Provide detailed analysis and concrete improvement suggestions."
                },
                {
                    "role": "user",
                    "content": f"""Calculate the match between this resume and job description:

Resume Data:
{json.dumps(resume_data, indent=2)}

Job Requirements:
{json.dumps(job_data, indent=2)}

Return a JSON object with:
{{
    "match_score": "Overall match percentage (0-100)",
    "confidence_score": "Analysis confidence level (0-100)",
    "skills_analysis": [
        {{
            "skill": "Name of required skill",
            "status": "found/missing/partial",
            "found_in_resume": "Where/how skill appears in resume",
            "relevance_score": "Relevance score (0-100)"
        }}
    ],
    "detailed_analysis": "Detailed analysis of the match",
    "improvement_suggestions": ["Specific suggestions to improve match"]
}}"""
                }
            ]

            response_text = self._get_llm_response(messages)
            clean_json = extract_json_from_text(response_text)
            return json.loads(clean_json)
            
        except Exception as e:
            logger.error(f"Match calculation error: {e}")
            return {
                "error": str(e),
                "timestamp": self.current_time,
                "match_score": 0,
                "confidence_score": 0,
                "skills_analysis": [],
                "detailed_analysis": f"Error: {str(e)}",
                "improvement_suggestions": ["An error occurred during analysis"]
            }
        
    def generate_cover_letter(self, resume_data: dict, job_data: dict) -> str:
        """Generate a personalized cover letter"""
        try:
            messages = [
                {
                "role": "system",
                "content": "You are a career coach writing customized cover letters."
                },
                {
                "role": "user",
                "content": f"""
Based on the following resume and job description, generate a compelling, personalized cover letter.

Resume:
{json.dumps(resume_data, indent=2)}

Job Description:
{json.dumps(job_data, indent=2)}

Make sure to:
- Mention the candidate's name and relevant experience
- Relate skills to job responsibilities
- End with a call to action
- Avoid placeholders
"""
            }
        ]
            return self._get_llm_response(messages).strip()
        except Exception as e:
            logger.error(f"Cover letter generation error: {e}")
            return f"Error generating cover letter: {str(e)}"

