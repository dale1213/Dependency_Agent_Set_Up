# GitHub Repository Analysis Tool

This tool automatically analyzes GitHub repositories by extracting command-line instructions from their READMEs, executing those commands, and analyzing the execution results using OpenAI’s large language model.

## Environment Setup

### Prerequisites
- Python 3.8+
- Git
- OpenAI API key

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create an Environment Variable File**:
   In the root directory of the project, create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   **Note:** Make sure to protect your API key and do not commit it to version control.

## Code Structure

### Main Components

- **LoopPipeline.py**: Core processing module
  - Clones GitHub repositories
  - Parses README files
  - Extracts and executes command-line instructions
  - Uses LLM to analyze execution results
  - Handles timeouts and errors
  
- **plot.py**: Results visualization module
  - Generates statistical charts from result files
  - Displays success/failure ratios and common error types
  
- **Data Files**:
  - `repos_data.json`: List of GitHub repositories to be analyzed
  - `results_larger_set.json`: Analysis results
  - `results_larger_set_statistics.json`: Statistical data

- **Output Directories**:
  - `chat_history/`: Stores LLM analysis chat logs
  - `command_history/`: Stores command execution histories

### Core Functions

- **Repository Processing**:
  - `run_from_github()`: Clones and processes a repository from a GitHub URL
  - `process_repos()`: Processes multiple repositories in parallel
  
- **Command Execution**:
  - `execute_and_analyze_command()`: Executes commands and displays real-time output
  - `explore_and_verify_setup()`: Explores repository setup and verifies the environment
  
- **LLM Analysis**:
  - `call_llm()`: Calls the OpenAI API for analysis
  - Provides detailed feedback on command execution results

## How to Use

1. **Analyze a Single Repository**:
   ```python
   from LoopPipeline import process_single_repo
   
   url = "https://github.com/user/repo"
   result = process_single_repo(url)
   ```

2. **Batch Process Multiple Repositories**:
   ```python
   from LoopPipeline import process_repos
   import json
   
   # Load the repository list
   with open("LLM_repo/repo_set/repos_data.json", "r") as f:
       repos = json.load(f)
   
   # Process repositories
   results = process_repos(repos)
   ```

3. **Generate Visualization of Results**:
   ```python
   from plot import generate_plots
   
   generate_plots()
   ```

## Real-Time Feedback Features

The tool provides real-time feedback during command execution:
- Command outputs are displayed in real time, labeled as `[OUT]` and `[ERR]`
- A progress bar shows the progress of command execution
- Timeout handling prevents commands from running indefinitely
- Detailed error reports and analysis are provided

## Notes

- Ensure that long-running commands have sufficient timeout settings.
- Some commands might require interactive input, which may not be automatically handled.
- Command execution occurs in a sandboxed environment, so ensure that no dangerous operations are included.