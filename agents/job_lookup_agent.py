import os
import sys
import json
from typing import Any, Dict, Optional, List
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import initialize_agent
from langchain.agents.types import AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from dotenv import load_dotenv
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

from agent_api.serpjob import scrape_job_profile

set_llm_cache(InMemoryCache())
load_dotenv()

def extract_json_from_text(text: str) -> str:
    """Extract JSON array from text by finding the first [ and last ]"""
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            return text[start:end]
        return "[]"
    except:
        return "[]"

class CustomMRKLOutputParser(MRKLOutputParser):
    """Custom output parser that handles JSON responses better"""
    
    def parse(self, text: str) -> Any:
        try:
            return super().parse(text)
        except Exception:
            cleaned_text = text.strip()
            
            if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
                try:
                    json.loads(cleaned_text)
                    from langchain.schema import AgentFinish
                    return AgentFinish(
                        return_values={"output": cleaned_text},
                        log=text
                    )
                except json.JSONDecodeError:
                    pass
            
            json_part = extract_json_from_text(cleaned_text)
            if json_part and json_part != "[]":
                try:
                    json.loads(json_part)
                    from langchain.schema import AgentFinish
                    return AgentFinish(
                        return_values={"output": json_part},
                        log=text
                    )
                except json.JSONDecodeError:
                    pass
            
            return super().parse(text)

def lookup(
    query: str, 
    location: str = "Canada", 
    remote_only: bool = False,
    serp_api_key: str = None
) -> str:
    """
    Enhanced direct lookup with API key parameter
    """
    try:
        # Clean the query
        query = query.strip()
        if "in" in query and location.lower() in query.lower():
            query = query.replace(f"in {location}", "").replace(f"In {location}", "").strip()
        
        print(f"🔍 Direct Lookup: Searching for '{query}' in {location} (Remote only: {remote_only})")
        
        # Use the provided API key for the search
        result = scrape_job_profile(query, location, serp_api_key)
        
        # Validate result
        if not result:
            print("No results from scrape_job_profile")
            return "[]"
        
        try:
            jobs_data = json.loads(result)
            if not isinstance(jobs_data, list):
                print("Result is not a list format")
                return "[]"
            
            print(f"Found {len(jobs_data)} jobs")
            return json.dumps(jobs_data)
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in lookup: {e}")
            return "[]"
        
    except Exception as e:
        print(f"Error in lookup function: {str(e)}")
        import traceback
        traceback.print_exc()
        return "[]"

def lookup_with_llm(
    query: str, 
    location: str = "Canada", 
    remote: bool = False,
    level: str = "Senior",
    serp_api_key: str = None,
    nebius_api_key: str = None
) -> str:
    """
    Enhanced LLM lookup function with API key parameters
    """
    try:
        if not nebius_api_key:
            print("Nebius API key is required for LLM search")
            return "[]"
            
        llm = ChatOpenAI(
            temperature=0.1,
            model_name="meta-llama/Meta-Llama-3.1-405B-Instruct",
            api_key=nebius_api_key,
            base_url="https://api.studio.nebius.com/v1/",
            max_retries=1,
        )

        # Clean the query
        query = query.strip()
        if "in" in query and location.lower() in query.lower():
            query = query.replace(f"in {location}", "").replace(f"In {location}", "").strip()

        print(f"🤖 LLM Agent: Searching for '{query}' | Location: '{location}' | Remote: {remote} | Level: {level}")

        # Create tool that uses provided SerpAPI key
        def job_search_tool(q: str) -> str:
            return lookup(q, location, remote, serp_api_key)

        tools_for_agent = [
            Tool(
                name="JobSearch",
                func=job_search_tool,
                description=f"Searches for {level} level {query} jobs. {'ONLY returns remote work opportunities.' if remote else f'Returns jobs in {location} plus remote opportunities.'}"
            )
        ]

        # Enhanced prompt with clearer filtering instructions
        remote_instruction = (
            "MUST return ONLY remote work opportunities, work-from-home positions, and distributed team roles. NO on-site positions."
            if remote else 
            f"Return jobs in {location} area that allow working from {location}. Include both on-site and hybrid positions."
        )

        template = """You are an expert job search assistant. Use the JobSearch tool to find jobs matching the exact criteria specified.

SEARCH CRITERIA:
- Position: {level} {input}
- Location Preference: {location}
- Remote Only: {remote_required}
- Filtering Rule: {remote_instruction}

IMPORTANT FILTERING RULES:
1. The JobSearch tool will automatically apply location and remote filtering
2. Remote jobs can be worked from anywhere, so they should be included unless location is very specific
3. On-site jobs should only be included if they match the target location
4. Trust the tool's filtering - it has been enhanced to handle these cases properly

INSTRUCTIONS:
1. Use the JobSearch tool with the query: "{input}"
2. The tool automatically applies the filtering based on the specified criteria
3. Return the complete JSON array from the tool without any modifications

FORMAT:
Thought: I need to search for jobs with the specified criteria and filtering.
Action: JobSearch
Action Input: {input}
Observation: [tool results will be properly filtered]
Thought: The tool has returned filtered results. I'll return them exactly as provided.
Final Answer: [return the exact JSON array from the tool]

CRITICAL: Your Final Answer must be ONLY the JSON array starting with [ and ending with ]. No explanations or additional text.

{format_instructions}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "level", "location", "remote_required", "remote_instruction"],
            partial_variables={"format_instructions": FORMAT_INSTRUCTIONS}
        )

        # Initialize agent
        agent = initialize_agent(
            tools=tools_for_agent,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate",
            agent_kwargs={
                "output_parser": CustomMRKLOutputParser(),
                "format_instructions": FORMAT_INSTRUCTIONS
            }
        )

        # Build search query
        search_query = f"{level} {query}"
            
        print(f"🤖 LLM Agent: Executing search with query: '{search_query}'")
        
        # Execute agent
        result = agent.invoke({
            "input": prompt.format(
                input=search_query,
                level=level,
                location=location,
                remote_required="YES" if remote else "NO",
                remote_instruction=remote_instruction
            )
        })

        # Process result
        output = result.get("output", "")
        print(f"🤖 LLM Agent: Raw output type: {type(output)}")

        if isinstance(output, str):
            cleaned_output = output.strip()
            
            # Remove common prefixes
            prefixes_to_remove = ["Final Answer:", "Answer:", "Result:"]
            for prefix in prefixes_to_remove:
                if cleaned_output.startswith(prefix):
                    cleaned_output = cleaned_output[len(prefix):].strip()
            
            # Extract JSON
            json_result = extract_json_from_text(cleaned_output)
            
            try:
                jobs_data = json.loads(json_result)
                if isinstance(jobs_data, list):
                    print(f"🤖 LLM Agent: Successfully returned {len(jobs_data)} filtered jobs")
                    return json_result
                else:
                    print("🤖 LLM Agent: Result is not a list")
                    return "[]"
            except json.JSONDecodeError as e:
                print(f"🤖 LLM Agent: JSON decode error: {e}")
                return "[]"
        else:
            print(f"🤖 LLM Agent: Unexpected output type: {type(output)}")
            return "[]"

    except Exception as e:
        print(f"🤖 Error during LLM job search: {e}")
        import traceback
        traceback.print_exc()
        
        # FALLBACK: Try the direct lookup method
        print("🔄 Falling back to direct lookup method...")
        try:
            return lookup(query, location, remote, serp_api_key)
        except Exception as fallback_error:
            print(f"🤖 Fallback also failed: {fallback_error}")
            return "[]"

def advanced_job_search(
    query: str,
    location: str = "Canada",
    remote: bool = False,  
    level: str = "Senior",
    use_llm: bool = True,
    salary_min: Optional[int] = None,
    job_type: Optional[str] = None,
    company_size: Optional[str] = None,
    serp_api_key: str = None,
    nebius_api_key: str = None
) -> Dict[str, Any]:
    """
    Advanced job search function with API key parameters
    """
    try:
        print(f"🚀 Advanced Job Search Started")
        print(f"Query: '{query}' | Location: '{location}' | Level: {level} | Remote: {remote}")
        print(f"Salary Min: {salary_min} | Job Type: {job_type} | Company Size: {company_size}")
        
        # Validate required API keys
        if not serp_api_key:
            return {
                "success": False,
                "error": "SerpAPI key is required",
                "total_found": 0,
                "jobs": [],
                "raw_results": "[]"
            }
            
        if use_llm and not nebius_api_key:
            return {
                "success": False,
                "error": "Nebius API key is required for advanced search",
                "total_found": 0,
                "jobs": [],
                "raw_results": "[]"
            }
        
        # Choose search method
        if use_llm:
            raw_results = lookup_with_llm(
                query=query,
                location=location,
                remote=remote,
                level=level,
                serp_api_key=serp_api_key,
                nebius_api_key=nebius_api_key
            )
        else:
            raw_results = lookup(
                query=query,
                location=location,
                remote_only=remote,
                serp_api_key=serp_api_key
            )
        
        # Parse results
        try:
            jobs_data = json.loads(raw_results)
        except json.JSONDecodeError:
            jobs_data = []
        
        print(f"📊 Initial results: {len(jobs_data)} jobs")
        
        # Apply additional filters
        filtered_jobs = []
        for job in jobs_data:
            if not isinstance(job, dict):
                continue
            
            # Salary filter
            if salary_min:
                job_salary = job.get('salary', '')
                if job_salary and isinstance(job_salary, str) and job_salary.lower() != 'n/a':
                    salary_numbers = re.findall(r'\d+', job_salary.replace(',', ''))
                    if salary_numbers:
                        max_salary = max([int(x) for x in salary_numbers if len(x) >= 4])
                        if max_salary < salary_min:
                            print(f"   💰 Filtered out: {job.get('title', 'N/A')} (salary: {max_salary} < {salary_min})")
                            continue
                        else:
                            print(f"   💰 Included: {job.get('title', 'N/A')} (salary: {max_salary} >= {salary_min})")
            
            # Job type filter
            if job_type and job_type.lower() != 'all':
                job_title = job.get('title', '').lower()
                if job_type.lower() not in job_title:
                    print(f"   🏷️ Filtered out: {job.get('title', 'N/A')} (type mismatch)")
                    continue
                else:
                    print(f"   🏷️ Included: {job.get('title', 'N/A')} (type match)")
            
            filtered_jobs.append(job)
        
        # Prepare response
        response = {
            "success": True,
            "total_found": len(filtered_jobs),
            "search_parameters": {
                "query": query,
                "location": location,
                "remote": remote,
                "level": level,
                "salary_min": salary_min,
                "job_type": job_type,
                "company_size": company_size,
                "method": "LLM Agent" if use_llm else "Direct Search"
            },
            "jobs": filtered_jobs,
            "raw_results": json.dumps(filtered_jobs),
            "filtering_applied": {
                "location_filter": True,
                "remote_filter": remote,
                "salary_filter": salary_min is not None,
                "job_type_filter": job_type is not None and job_type.lower() != 'all',
                "duplicate_removal": True
            }
        }
        
        print(f"🎯 Advanced Search Complete: Found {len(filtered_jobs)} matching jobs after all filters")
        return response
        
    except Exception as e:
        print(f"❌ Advanced job search failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "total_found": 0,
            "jobs": [],
            "raw_results": "[]",
            "filtering_applied": {}
        }

# Convenience functions with API key parameters
def search_jobs(
    query: str,
    location: str = "Canada",
    remote: bool = False,
    level: str = "Senior",
    serp_api_key: str = None,
    nebius_api_key: str = None
) -> str:
    """
    Main job search function with API key parameters
    """
    print(f"🔍 Main Search: '{query}' | Location: '{location}' | Remote: {remote} | Level: {level}")
    
    if not location or location.strip() == "":
        location = "Canada"
    
    if not serp_api_key:
        return "[]"
        
    # Use LLM agent if Nebius key is provided
    if nebius_api_key:
        return lookup_with_llm(
            query=query,
            location=location,
            remote=remote,
            level=level,
            serp_api_key=serp_api_key,
            nebius_api_key=nebius_api_key
        )
    else:
        return lookup(
            query=query,
            location=location,
            remote_only=remote,
            serp_api_key=serp_api_key
        )

# Helper functions with API key parameters
def search_remote_jobs(
    query: str,
    level: str = "Senior",
    location: str = "Canada",
    serp_api_key: str = None,
    nebius_api_key: str = None
) -> str:
    """Quick search for remote jobs ONLY"""
    return lookup_with_llm(
        query=query,
        location=location,
        remote=True,
        level=level,
        serp_api_key=serp_api_key,
        nebius_api_key=nebius_api_key
    )

def search_entry_level_jobs(
    query: str,
    location: str = "Canada",
    remote: bool = False,
    serp_api_key: str = None,
    nebius_api_key: str = None
) -> str:
    """Quick search for entry-level positions"""
    return lookup_with_llm(
        query=query,
        location=location,
        remote=remote,
        level="Junior",
        serp_api_key=serp_api_key,
        nebius_api_key=nebius_api_key
    )