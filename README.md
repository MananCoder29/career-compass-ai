---
title: Career Compass AI
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
tags:
  - agent-demo-track
---
# Career Compass AI 🧭

An AI-powered career assistant that combines intelligent job searching, resume matching, and cover letter generation using an agent-based architecture.

# Demo Link 📹
[https://youtu.be/7jiuIUsO28k]

## 🌟 Key Features

### 1. 🔍 Smart Job Search
- AI-enhanced job discovery using LangChain agents
- SerpAPI integration for comprehensive job data
- Intelligent filtering for location and remote work
- Salary information parsing when available
- Experience level filtering
- Support for multiple locations including US, Canada, UK, and more
- Fast and Advanced search modes

### 2. 📊 Resume Analysis & Matching
- PDF resume parsing and analysis
- AI-powered job description matching
- Detailed skills gap analysis
- Match score calculation
- Confidence metrics
- Improvement suggestions

### 3. ✍️ Cover Letter Generation
- Context-aware content generation
- Personalized based on resume and job
- Professional formatting
- Customizable output

## 🤖 Agent Architecture

### Job Search Agent
- Uses LangChain for orchestration
- Custom MRKL output parser
- Intelligent retry mechanisms
- Contextual job filtering

### Resume Analysis Agent
- Resume parsing capabilities
- Skills extraction
- Match calculation
- Improvement suggestions

## 🎨 Custom UI Components
- Modern, responsive design
- Dark mode support
- Interactive job listings table
- Analysis dashboards
- Status indicators
- Export functionality (CSV/JSON)

## 🔑 Required API Keys

1. **SerpAPI Key** (Required for all searches)
   - Get from [SerpAPI](https://serpapi.com)
   - Used for job searching

2. **Nebius API Key** (Required for Advanced Search)
   - Get from [Nebius](https://nebius.ai)
   - Powers AI-enhanced features

## 📝 How to Use

1. Configure API Keys
2. Choose search method:
   - Advanced Search (AI-Enhanced, 30-60s)
   - Basic Search (Fast, 10-30s)
3. Enter job search criteria:
   - Job title/keywords
   - Location
   - Experience level
   - Remote work preference
4. View results and export if needed
5. For resume matching:
   - Upload PDF resume
   - Paste job description
   - Get detailed analysis
   - Generate custom cover letter

## 🔒 Security
- API keys stored in memory only
- No permanent data storage
- Secure data handling
- Session-based isolation

## 📊 Session Information
- Current Time: 2025-06-10 18:47:07 UTC
- Current User: mananshah296
- Version: 2.0.0
- Last Updated: 2025-06-11

## 🚀 Try Live Demo 👇
- [click here](https://huggingface.co/spaces/Agents-MCP-Hackathon/job-hunting-ai)

## 🏗️ High-Level Architecture Diagram
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6765bc297fe8b213e581fda0/YP0H5wlTC9G0Oqn7vKwUH.png)
