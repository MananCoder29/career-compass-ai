# agent_api/serpjob.py 
import os
from dotenv import load_dotenv
import json
from serpapi import GoogleSearch
import time
from typing import List, Dict, Optional, Tuple

load_dotenv()

def test_api_connection(api_key: str = None) -> Tuple[bool, str]:
    """Test SerpAPI connection with provided key"""
    try:
        if not api_key:
            return False, "API key is required"
        
        # Test with minimal query
        params = {
            "api_key": api_key,
            "engine": "google",
            "q": "test",
            "num": 1
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return False, f"API Error: {results['error']}"
            
        return True, "Connection successful"
        
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def clean_salary_text(text: str) -> str:
    """Clean and format salary information"""
    if not text or text.lower() in ['n/a', 'not specified', '']:
        return "Not specified"
    
    # Clean whitespace
    cleaned = ' '.join(text.split())
    
    # Format common salary terms
    replacements = {
        'a year': '/year',
        'an hour': '/hour', 
        'a month': '/month',
        'per year': '/year',
        'per hour': '/hour',
        'per month': '/month'
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    return cleaned

def is_job_remote(job: Dict) -> bool:
    """Determine if a job is remote based on various indicators"""
    
    # Collect all text fields to analyze
    text_sources = [
        job.get("title", ""),
        job.get("description", ""),
        job.get("location", ""),
        job.get("company_name", ""),
        str(job.get("detected_extensions", {}))
    ]
    
    # Join all text and convert to lowercase
    full_text = " ".join(text_sources).lower()
    
    # Remote work indicators
    remote_keywords = [
        "remote", "work from home", "wfh", "virtual",
        "anywhere", "fully remote", "remote-first", "100% remote",
        "work remotely", "distributed team", "telecommute",
        "home-based", "remote position", "flexible location",
        "work from anywhere", "remote-friendly", "home office"
    ]
    
    return any(keyword in full_text for keyword in remote_keywords)

def scrape_job_profile(query: str, location: str = "Canada", api_key: str = None) -> str:
    """
    Scrape job information from Google Jobs using provided SerpAPI key
    """
    print(f"\n{'='*50}")
    print(f"🔍 STARTING JOB SEARCH")
    print(f"Query: {query}")
    print(f"Location: {location}")
    print(f"{'='*50}")
    
    try:
        # Validate API key
        if not api_key:
            print("No API key provided")
            return json.dumps([])
        
        # Test API connection first
        api_connected, api_message = test_api_connection(api_key)
        if not api_connected:
            print(f"API Connection Failed: {api_message}")
            return json.dumps([])
        
        print(f"API Connection: {api_message}")
        
        # Clean and prepare search query
        search_query = query.strip()
        
        # Add remote keyword if not present
        if "remote" not in search_query.lower():
            search_query = f"{search_query} remote"
        
        # Location configuration mapping
        location_settings = {
            "United States": {"location": "United States", "gl": "us", "hl": "en"},
            "Canada": {"location": "Canada", "gl": "ca", "hl": "en"},
            "United Kingdom": {"location": "United Kingdom", "gl": "gb", "hl": "en"},
            "Australia": {"location": "Australia", "gl": "au", "hl": "en"},
            "Germany": {"location": "Germany", "gl": "de", "hl": "en"},
            "Netherlands": {"location": "Netherlands", "gl": "nl", "hl": "en"},
            "Remote Worldwide": {"location": "", "gl": "us", "hl": "en"}
        }
        
        # Get location settings
        loc_config = location_settings.get(location, location_settings["Canada"])
        
        # Add location to query if not worldwide
        if location != "Remote Worldwide" and loc_config["location"]:
            search_query = f"{search_query} in {loc_config['location']}"
        
        print(f"📝 Final search query: '{search_query}'")
        
        # Build search parameters
        search_params = {
            "api_key": api_key,
            "engine": "google_jobs",
            "q": search_query,
            "hl": loc_config["hl"],
            "gl": loc_config["gl"],
            "chips": "date_posted:month",
            "num": 12  # Reasonable number of results
        }
        
        # Add location parameter if specified
        if loc_config["location"]:
            search_params["location"] = loc_config["location"]
        
        print(f"🔧 Search parameters:")
        for key, value in search_params.items():
            if key != "api_key":  # Don't log API key
                print(f"   {key}: {value}")
        
        # Execute search with retry mechanism
        max_attempts = 1
        results = None
        
        for attempt in range(max_attempts):
            try:
                print(f"Search attempt {attempt + 1}/{max_attempts}")
                
                search = GoogleSearch(search_params)
                results = search.get_dict()
                
                # Check for API errors
                if "error" in results:
                    error_msg = results["error"]
                    print(f"API Error on attempt {attempt + 1}: {error_msg}")
                    
                    if attempt < max_attempts - 1:
                        print("⏳ Waiting before retry...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print("All attempts failed")
                        return json.dumps([])
                
                # Success
                print(f"Search successful on attempt {attempt + 1}")
                break
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    print("⏳ Waiting before retry...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print("All search attempts exhausted")
                    return json.dumps([])
        
        # Extract job results
        raw_jobs = results.get("jobs_results", [])
        print(f"📊 Raw jobs found: {len(raw_jobs)}")
        
        if not raw_jobs:
            print("No jobs found in search results")
            return json.dumps([])
        
        # Process and filter jobs
        processed_jobs = []
        
        for idx, job in enumerate(raw_jobs):
            try:
                print(f"\n Processing job {idx + 1}: {job.get('title', 'Unknown Title')}")
                
                # Extract basic information
                title = job.get("title", "N/A")
                company = job.get("company_name", "N/A")
                job_location = job.get("location", "N/A")
                
                print(f"   Company: {company}")
                print(f"   Location: {job_location}")
                
                # Extract salary information
                salary = "Not specified"
                if job.get("salary_snippet"):
                    salary = clean_salary_text(job["salary_snippet"].get("text", ""))
                elif job.get("detected_extensions", {}).get("salary"):
                    salary = clean_salary_text(job["detected_extensions"]["salary"])
                
                print(f"   Salary: {salary}")
                
                # Determine if job is remote
                remote_status = is_job_remote(job)
                print(f"   Remote: {'Yes' if remote_status else 'No'}")
                
                # Extract apply link
                apply_link = "N/A"
                
                # Try multiple sources for apply link
                if job.get("apply_options") and len(job["apply_options"]) > 0:
                    apply_link = job["apply_options"][0].get("link", "N/A")
                elif job.get("share_link"):
                    apply_link = job["share_link"]
                elif job.get("link"):
                    apply_link = job["link"]
                
                # Extract posting date
                posted_date = "Recently"
                if job.get("detected_extensions", {}).get("posted_at"):
                    posted_date = job["detected_extensions"]["posted_at"]
                elif job.get("posted_at"):
                    posted_date = job["posted_at"]
                
                # Create job entry
                job_entry = {
                    "title": title,
                    "company_name": company,
                    "location": job_location,
                    "salary": salary,
                    "remote": "Yes" if remote_status else "No",
                    "posted_at": posted_date,
                    "link": apply_link
                }
                
                # Apply filtering logic
                should_include = False
                
                if location == "Remote Worldwide":
                    # For worldwide remote, only include remote jobs
                    should_include = remote_status
                    filter_reason = "remote status" if remote_status else "not remote"
                else:
                    # For specific locations, include if matches location OR is remote
                    location_match = location.lower() in job_location.lower()
                    should_include = location_match or remote_status
                    
                    if location_match and remote_status:
                        filter_reason = "location match + remote"
                    elif location_match:
                        filter_reason = "location match"
                    elif remote_status:
                        filter_reason = "remote job"
                    else:
                        filter_reason = "no match"
                
                if should_include:
                    processed_jobs.append(job_entry)
                    print(f" INCLUDED ({filter_reason})")
                else:
                    print(f" FILTERED OUT ({filter_reason})")
                
            except Exception as e:
                print(f"Error processing job {idx + 1}: {str(e)}")
                continue
        
        print(f"\n{'='*50}")
        print(f"📊 FINAL RESULTS")
        print(f"Raw jobs found: {len(raw_jobs)}")
        print(f"Jobs after filtering: {len(processed_jobs)}")
        print(f"{'='*50}")
        
        # Sort by posting date (attempt to put recent jobs first)
        try:
            processed_jobs.sort(
                key=lambda x: x.get("posted_at", ""),
                reverse=True
            )
        except:
            print("⚠️ Could not sort by posting date")
        
        # Return results as JSON
        result_json = json.dumps(processed_jobs, indent=2)
        print(f"Returning {len(processed_jobs)} jobs")
        
        return result_json
        
    except Exception as e:
        print(f"\n CRITICAL ERROR in scrape_job_profile:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return json.dumps([])

def quick_test(api_key: str = None):
    """Quick test function to verify the scraper works"""
    print("🧪 TESTING SCRAPER")
    print("="*30)
    
    # Test connection first
    connected, message = test_api_connection(api_key)
    print(f"Connection test: {message}")
    
    if not connected:
        return False
    
    # Test search
    try:
        result = scrape_job_profile("Python developer", "Canada", api_key)
        jobs = json.loads(result)
        
        print(f"Test completed: Found {len(jobs)} jobs")
        if jobs:
            sample_job = jobs[0]
            print(f"Sample job: {sample_job['title']} at {sample_job['company_name']}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

# if __name__ == "__main__":
#     # Only run test if API key is provided
#     api_key = os.environ.get("SERP_API_KEY")
#     if api_key:
#         quick_test(api_key)
#     else:
#         print("No API key found in environment variables")