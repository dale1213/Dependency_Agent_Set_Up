import os
import zipfile
import subprocess
import tempfile
import shutil
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
import json
import pandas as pd
import logging
import re
import tiktoken 
import time
import uuid


from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from context.RepositoryContext import RepositoryContext


logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
print("Starting script execution...", flush=True)  # Ensure this prints

# Load API key and instantiate OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Ensure it is set in the .env file.")

client = OpenAI(api_key=api_key)

MAX_MESSAGE_LENGTH = 1048000

##############################
# LLM helper functions
##############################
def truncate_text_middle(text, max_length=MAX_MESSAGE_LENGTH, skip_marker="[Skipped due to length constraints]"):
    """
    If the text length exceeds max_length, truncate the text in the middle,
    """
    if len(text) <= max_length:
        return text
    marker_length = len(skip_marker)
    # The remaining characters that can be kept
    keep_length = max_length - marker_length
    # Keep half of the characters from the head and the tail
    head_length = keep_length // 2
    tail_length = keep_length - head_length
    return text[:head_length] + skip_marker + text[-tail_length:]

def call_llm(messages, system_message=None, model="gpt-3.5-turbo", context: RepositoryContext = None):
    """
    Call the LLM model with given messages and optional context.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys
        model (str, optional): The LLM model to use. Defaults to "gpt-3.5-turbo"
        context (RepositoryContext, optional): Repository context object to store chat history. Defaults to None
        
    Returns:
        str: The LLM response text
    """
    logger.info("Sending LLM request")
    
    # Combine chat history with new messages if context exists
    if context:
        # Get the new user message if it exists
        new_message = next((msg for msg in messages if msg['role'] == 'user'), None)
        # Generate messages using context, including chat history and new message
        # Use a safe limit for total message length (leave room for response)
        max_total_length = MAX_MESSAGE_LENGTH * 0.9  # 90% of the limit to leave room for response
        messages = context.generate_messages(
            message=new_message, 
            system_message=system_message,
            max_total_length=max_total_length
        )
    else:
        # Only truncate messages if we don't have a context object
        for message in messages:
            message['content'] = truncate_text_middle(message['content'])
    
    # Send request to LLM
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    output = response.choices[0].message.content.strip()
    
    # Log the request and response
    if context:
        context.add_chat_message('assistant', output)
    
    return output

def prefilter_text(text):
    """
    Filtering of command-like lines. use it with caution: possibly failed to caputure uncommon commands
    - This method hasn't been used, but for the cost consideration of using LLM, just put it here in case we need it in the future
    
      - Merges multi-line commands that use a backslash (\) for line continuation.
      - Detects heredoc blocks with any delimiter.
      - Removes simple inline comments (anything after an unquoted #).
      - Captures inline variable assignments (allowing spaces around '=' and multiple assignments).
      - Recognizes subshell execution using both $(...) and backticks.
      - Detects alias definitions and function definitions (using both "function" and the shorthand name() {).
      - Captures control structures (if/elif/else/fi, for/while/until/do/done, case/esac) as blocks.
      - Captures command groups using braces { ... } or parentheses ( ... ).
      - Includes commands with pipes, logical operators, and redirections.
    """
    # First, merge lines that end with a backslash
    merged_lines = []
    buffer = ""
    for line in text.splitlines():
        # Remove trailing whitespace but keep indentation (for heredoc or block formatting)
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
        else:
            buffer += stripped
            merged_lines.append(buffer)
            buffer = ""
    if buffer:
        merged_lines.append(buffer)

    # Remove simple inline comments (this is simplistic and may remove '#' in strings)
    def remove_inline_comments(line):
        # This naive approach splits on '#' if it is preceded by whitespace
        # For a robust solution, a proper shell parser would be needed.
        return re.split(r'\s+#', line, maxsplit=1)[0].strip()

    merged_lines = [remove_inline_comments(line) for line in merged_lines if line.strip()]

    filtered_lines = []
    heredoc_buffer = []
    in_heredoc = False
    heredoc_delimiter = None

    multi_line_block = []
    in_block = False
    # End tokens for control blocks including if/for/while/until/case
    block_end_tokens = {"fi", "done", "esac"}

    # Regex to capture inline variable assignments allowing extra spaces and multiple assignments.
    inline_assignment_pattern = re.compile(r"^(?:\w+\s*=\s*\S+\s+)+\S+")
    # Regex for function definitions: either function keyword or the shorthand pattern.
    function_pattern = re.compile(r"^(?:function\s+\w+\s*\{|[\w\-_]+\s*\(\)\s*\{)")
    # Regex for detecting command groups with braces or parentheses at the start.
    group_pattern = re.compile(r"^[\{\(].+[\}\)]\s*$")
    # Keywords for control structures that begin blocks (adding "until")
    block_start_keywords = re.compile(r"^(if|for|while|until|case|elif|else)\b")

    # Common execution keywords (a wide set of common commands and builtins)
    execution_keywords = (
        "python", "./", "bash", "sh ", "make", "npm", "yarn", "pip",
        "git", "docker", "gcc", "java", "go ", "node", "cargo", "ruby", "perl",
        "mvn", "gradle", "rustc", "flutter", "dotnet", "kubectl", "helm", "conda",
        "eval", "exec", "nohup", "trap", "xargs", "alias"
    )

    # Process each merged line.
    for line in merged_lines:
        line = line.strip()

        # Heredoc handling: detect start, then capture until a line exactly equals the delimiter.
        heredoc_match = re.search(r"<<\s*(\S+)", line)
        if heredoc_match and not in_heredoc:
            in_heredoc = True
            heredoc_delimiter = heredoc_match.group(1)
            heredoc_buffer.append(line)
            continue
        if in_heredoc:
            heredoc_buffer.append(line)
            # End the heredoc if the line (after stripping) equals the delimiter.
            if line.strip() == heredoc_delimiter:
                filtered_lines.extend(heredoc_buffer)
                heredoc_buffer = []
                in_heredoc = False
                heredoc_delimiter = None
            continue

        # Skip lines that are empty after comment removal
        if not line:
            continue

        # Handle multi-line control structures as blocks.
        if block_start_keywords.match(line) or line.endswith("do") or line.endswith("then"):
            in_block = True
            multi_line_block.append(line)
            continue
        if in_block:
            multi_line_block.append(line)
            # If the line exactly matches an end token, end the block.
            if line in block_end_tokens:
                filtered_lines.extend(multi_line_block)
                multi_line_block = []
                in_block = False
            continue

        # Capture inline variable assignments (even multiple assignments)
        if inline_assignment_pattern.match(line):
            filtered_lines.append(line)
            continue

        # Capture subshell executions: both $() and backticks.
        if "$(" in line or "`" in line:
            filtered_lines.append(line)
            continue

        # Capture alias definitions and function definitions.
        if line.startswith("alias ") or function_pattern.match(line):
            filtered_lines.append(line)
            continue

        # Capture command groups enclosed in braces or parentheses.
        if group_pattern.match(line):
            filtered_lines.append(line)
            continue

        # Capture lines that include any execution keywords.
        if any(tok in line for tok in execution_keywords):
            filtered_lines.append(line)
            continue

        # Capture general command lines that start with a word (including common commands like ls, pwd, etc.)
        if re.match(r"^[a-zA-Z0-9\-_]+\s", line):
            filtered_lines.append(line)
            continue

        # Capture lines with pipes, logical operators, or redirection operators.
        if any(op in line for op in ("|", "&&", ";", ">", ">>", "<")):
            filtered_lines.append(line)
            continue

    return "\n".join(filtered_lines)


def split_readme_into_chunks(README, max_token_limit=16000):
    """
    Splits the README into chunks while ensuring that entire lines (commands or text) remain intact.
    - group full lines into chunks without exceeding max_token_limit.
    - Ensures multi-line commands using '\' are merged before splitting.
    - Avoids breaking a line into separate chunks.
    """
    enc = tiktoken.get_encoding("cl100k_base") 

    # Merge multi-line commands (backslash `\` continuation)
    merged_lines = []
    buffer = ""

    for line in README.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):  # If the line ends with '\', merge it
            buffer += stripped[:-1] + " "  # Remove '\' and add space
        else:
            buffer += stripped
            merged_lines.append(buffer)
            buffer = ""

    if buffer:
        merged_lines.append(buffer)


    chunks = []
    current_chunk = []
    current_token_count = 0

    for line in merged_lines:
        token_count = len(enc.encode(line)) 

        if current_token_count + token_count > max_token_limit:
            # If adding this line exceeds the limit, finalize the current chunk
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]  # Start a new chunk with this line
            current_token_count = token_count
        else:
            current_chunk.append(line)
            current_token_count += token_count

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

##############################
# Existing Functions (File-based)
##############################

def get_command_lines_from_readme(README, context: RepositoryContext = None):
    """
    Extracts command-line instructions from large README text by:
    - Splitting into chunks if necessary
    - Sending each chunk to the LLM separately
    - Concatenating the extracted commands
    """
    logger.info("Extracting command lines from provided README.")
    if not README:
        logger.warning("No README provided for command extraction.")
        return "No README provided."

    chunks = split_readme_into_chunks(README, max_token_limit=16300)
    extracted_commands = []

    for idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")

        prompt = f"""Extract only the valid, executable shell commands from the following README text. Follow these rules strictly:
            1. Return one valid shell command per line, without any additional commentary.
            2. Do not include any markdown formatting such as triple backticks, asterisks, or hyphens used for lists.
            3. Remove any extraneous characters, inline explanations, or documentation text.
            4. Preserve multi-line commands (using '\' for line continuation) and command sequences (with operators like && or ;).
            5. Output only commands that can be directly executed in a Unix-like shell.
            6. If a line does not represent a valid shell command (e.g., a link, descriptive text, or a markdown heading), skip it.

            For example, if the README contains:
                - `npm i -g @saleor/cli`
                - Some text: For installation, run npm i -g @saleor/cli`
            You should output:
            npm i -g @saleor/cli
            Now, extract the commands from the following text: {chunk}"""


        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            extracted_chunk = call_llm(messages, context=context)
            print("In get_command_lines_from_readme, the context['chat_history'] is: ", context.get_full_context()['chat_history'])
            cleaned_chunk = extracted_chunk.replace("```bash", "").replace("```", "").strip()
            extracted_commands.append(cleaned_chunk)
        except Exception as e:
            logger.error(f"Error processing chunk {idx + 1}: {str(e)}")
            continue

    final_commands = "\n".join(extracted_commands).strip()
    logger.info("Completed extraction of command lines from provided text.")
    logger.debug(f"Extracted commands: {final_commands[:200]}{'...' if len(final_commands) > 200 else ''}")

    return final_commands

def validate_json_response(response):
    """
    Validates that the response is a valid JSON with required fields.
    Returns the parsed JSON if valid, raises ValueError if not.
    """
    try:
        data = json.loads(response)
        required_fields = [
            "exploration_command",
            "executable_command",
            "verified_dependency_setup",
            "analysis"
        ]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return data
    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON")

def explore_and_verify_setup(repo_dir, context: RepositoryContext):
    """
    Let agent freely explore the repository and verify the setup through multiple interactions.
    
    Args:
        repo_dir: Repository directory path
        context: RepositoryContext object containing chat history
    
    Returns:
        bool: True if setup is verified
    """
    MAX_ATTEMPTS = 10  # Maximum number of exploration/verification attempts
    attempt_count = 0
    exploration_prompt =  """You MUST respond in the following JSON format ONLY. Any non-JSON response will be rejected:

        {
            "exploration_command": "next exploration command or null if ready to verify",
            "executable_command": "verification command or null if still exploring",
            "verified_dependency_setup": true/false,
            "analysis": "detailed explanation of:
                        - what you found
                        - why you chose this command
                        - what you plan to check next"
        }

        RULES:
        1. Your response MUST be valid JSON
        2. DO NOT include any text outside the JSON structure
        3. DO NOT include any explanations or markdown
        4. DO NOT use triple backticks
        5. The JSON must contain all fields shown above
        
        You can use these commands to explore:
        - ls [-la] [path]: List directory contents
        - pwd: Show current directory
        - cat [file]: View file contents
        - find . -name [pattern]: Search for files
        - python/python3 -c "import [package]": Test if a package is installed

        OBJECTIVES:
        1. Explore the repository structure
        2. Identify main executable files
        3. Check if required dependencies are installed
        4. Find and verify configuration files
        5. Locate test files or example scripts

        STRATEGY:
        1. Start with basic directory exploration (ls, pwd)
        2. Look for key files:
           - requirements.txt, setup.py, package.json
           - main executable files
           - configuration files
        3. Test dependencies if found
        4. Try to run tests or example scripts
        5. When confident, provide a verification command

        IMPORTANT:
        - Keep commands safe (no modifications)
        - One command per response
        - Set verified_dependency_setup=true only when you've confirmed everything works
    """
    

    while True:
        # Check attempt limit
        attempt_count += 1
        if attempt_count > MAX_ATTEMPTS:
            context.add_chat_message(
                "system",
                "Maximum verification attempts reached. Stopping verification process."
            )
            return False

        # Get agent's next action
        user_prompt = {
            "role": "user",
            "content": """You are an autonomous agent exploring a repository to verify its setup.
                    Based on your previous exploration results, determine your next action.

                    Return ONLY a valid JSON object with this structure:
                    {
                        "exploration_command": "your next shell command to explore, or null if ready to verify",
                        "executable_command": "your verification command if ready, or null if still exploring",
                        "verified_dependency_setup": boolean indicating if setup is verified,
                        "analysis": "your analysis of findings and next steps"
                    }

                    CRITICAL:
                    1. You are an agent, not a human assistant
                    2. Respond with pure JSON only
                    3. No explanations or text outside JSON
                    4. No markdown or formatting"""
        }
        
        try:
            response = call_llm(
                messages=[user_prompt],
                system_message=exploration_prompt,
                context=context
            )
            
            # Validate and parse the JSON response
            action = validate_json_response(response)
            
            # If agent wants to explore more
            if action["exploration_command"]:
                process = subprocess.Popen(
                    action["exploration_command"],
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=repo_dir
                )
                stdout, stderr = process.communicate(timeout=60)
                
                # Add the result to context
                context.add_chat_message(
                    "user",
                    f"Command output:\n{stdout}\n{stderr}"
                )
                continue
                
            # If agent found an executable to test
            if action["executable_command"]:
                process = subprocess.Popen(
                    action["executable_command"],
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=repo_dir
                )
                stdout, stderr = process.communicate(timeout=60)
                
                # Add the result to context
                context.add_chat_message(
                    "user",
                    f"""Verification command output:
                        Command: {action['executable_command']}
                        Output:
                        {stdout}
                        {stderr}"""
                )
                # Execute all executable commands and analyze results
                all_outputs = []
                for cmd in action["executable_command"].split(";"):
                    cmd = cmd.strip()
                    if not cmd:
                        continue
                        
                    process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=repo_dir
                    )
                    stdout, stderr = process.communicate(timeout=60)
                    all_outputs.append({
                        "command": cmd,
                        "stdout": stdout,
                        "stderr": stderr,
                        "return_code": process.returncode
                    })

                # Have LLM analyze the results
                analysis_prompt = {
                    "role": "user",
                    "content": f"""Please analyze the outputs from multiple verification commands and determine if dependencies are properly set up.
                    If there are any issues, explain what went wrong.
                    
                    Command outputs:
                    {json.dumps(all_outputs, indent=2)}
                    
                    Please respond with a JSON in this format:
                    {{
                        "success": true/false,
                        "analysis": "detailed explanation of what worked or what went wrong"
                    }}"""
                }
                
                analysis_response = call_llm(
                    messages=[analysis_prompt],
                    system_message="You are a helpful assistant analyzing command outputs to verify dependency setup.",
                    context=context
                )
                
                try:
                    analysis = json.loads(analysis_response)
                    print("The analysis is: ", analysis)
                    if not analysis["success"]:
                        return (False, analysis["analysis"])
                    else:
                        return (True, analysis["analysis"])
                except:
                    context.add_chat_message(
                        "assistant",
                        "Failed to parse analysis response"
                    )
                    return (False, analysis_response)
                
            # If agent couldn't find anything to verify
            if not action["exploration_command"] and not action["executable_command"]:
                return (False, "No exploration or executable command found")
                
        except ValueError as e:
            # If JSON validation fails, add error to context and retry
            context.add_chat_message(
                "user",
                f"Error: Invalid response format. Please provide valid JSON. Details: {str(e)}"
            )
            continue
        except Exception as e:
            context.add_chat_message(
                "user",
                f"Error during execution: {str(e)}"
            )
            return (False, str(e))

def run_ls(working_dir):
    """
    Runs 'ls -la' in the given working directory and returns the stdout output.
    """
    try:
        result = subprocess.run("ls -la", shell=True, capture_output=True, text=True, cwd=working_dir)
        return result.stdout
    except Exception as e:
        return f"Error running ls: {str(e)}"

def get_executable_command_from_ls(ls_output):
    """
    Uses ChatGPT to analyze the output of 'ls -la' and return the command(s) needed
    to run the primary executable file that can verify whether dependencies have been installed properly.
    """
    prompt = (
        "You are given the output of an 'ls -la' command executed in the root of a cloned repository:\n"
        f"{ls_output}\n\n"
        "Based on this listing, please provide the command-line instruction(s) that would run the primary executable "
        "or self-test of the application, which can help verify that all dependencies are properly installed. "
        "Only output the command(s) without any additional explanation or markdown formatting."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    command_output = response.choices[0].message.content.strip()
    # Remove any markdown formatting if present
    command_output = command_output.replace("```bash", "").replace("```", "").strip()
    return command_output

def run_executable(executable_command, working_dir):
    """
    Runs the given executable command in the specified working directory.
    Returns a generator that streams the execution log.
    """
    log_history = ""
    if not executable_command:
        log_history += "❌ No executable command provided.\n"
        yield log_history
        return

    log_history += f"Running executable command: {executable_command}\n"
    yield log_history

    try:
        process = subprocess.Popen(
            executable_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir
        )

        # Stream stdout line-by-line
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            log_history += line
            yield log_history

        stderr_output = process.stderr.read()
        if stderr_output:
            log_history += stderr_output
            log_history += "\n❌ Error while running the executable command.\n"
            yield log_history
            return

        exit_code = process.wait()
        if exit_code == 0:
            log_history += "\n✅ Executable command ran successfully.\n"
        else:
            log_history += f"\n❌ Executable command failed with exit code {exit_code}.\n"
        yield log_history

    except Exception as e:
        log_history += f"\n❌ Exception while running executable command: {str(e)}\n"
        yield log_history



##############################
# New Functions for GitHub URL-based Execution
##############################

def clone_repo(github_url):
    """
    Clones the GitHub repository into a directory under /home/ec2-user.
    Returns the path to the cloned repository.
    """
    base_dir = "/home/ec2-user/repo_temp/repos"
    os.makedirs(base_dir, exist_ok=True)
    
    repo_name = github_url.split('/')[-1]
    unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    repo_dir = os.path.join(base_dir, f"{repo_name}_{unique_id}")
    
    cmd = f"git clone {github_url} {repo_dir}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        raise Exception(f"Failed to clone repository: {result.stderr}")
    
    return repo_dir

def get_command_lines_from_text(text):
    """
    Uses GPT to extract command-line instructions from a given text.
    """
    if not text:
        return "No text provided."
    prompt = (
        "You are given the contents of a README file below. "
        "Please extract and print only the command-line instructions. "
        "Ignore all other text. Remove triple backticks, etc.\n\n"
        f"{text}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    extracted_commands = response.choices[0].message.content
    cleaned_commands = extracted_commands.replace("```bash", "").replace("```", "").strip()
    return cleaned_commands

def execute_and_analyze_command(command, repo_dir, context: RepositoryContext):
    """
    Execute a single command and let GPT analyze the output.
    """
    log_output = ""
    try:
        
        process = subprocess.Popen(
            command, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo_dir
        )
        stdout, stderr = process.communicate(timeout=300)
        log_output = stdout + stderr
        print("The log output is: ", log_output)
        if process.returncode != 0:
            print(f"Command failed with return code: {process.returncode}")
            
        # Update system message in context
        system_prompt = """You are a helpful assistant that analyzes command outputs and suggests next steps.
            When analyzing command failures:
            1. For directory operations (cd, ls, etc):
            - First try to list directory contents
            - Then suggest creating directory if needed
            2. For port conflicts:
            - Suggest using a different port
            - Or provide command to kill existing process
            3. Mark as critical_failure only if:
            - Required files are missing and can't be created
            - Dependencies can't be installed
            - System resources are unavailable
            4. Mark dependency_setup as true ONLY when ALL of these are completed:
            - Virtual environment is created AND activated (if needed)
            - ALL required packages are installed (pip, conda, npm, etc.)
            - ALL configuration files are in place
            - No remaining dependency-related commands in the instruction list
            
            IMPORTANT: dependency_setup must be false if:
            - There are any remaining package installation commands (pip, conda, etc.)
            - Any installation command failed
            - Not all commands in the original command list have been executed
            - The final verification command hasn't been run successfully
            """
        
        # Create user message
        user_message = {
            "role": "user",
            "content": f"""
                Analyze this command output and suggest the next command to run:
                Command executed: {command}
                Output:
                {log_output}

                Respond in this JSON format:
                {{
                    "success": true/false,
                    "critical_failure": true/false,
                    "dependency_setup": true/false,
                    "analysis": "brief analysis of what happened",
                    "next_command": "executable shell command or null if no further action needed",
                    "alternative_command": "executable shell command or null if no alternative needed"
                }}

                Remember: 
                1. For next_command and alternative_command, only provide actual executable shell commands, not descriptions.
                2. Set dependency_setup to true only when all dependencies are properly installed and environment is ready.

                Example of good responses:
                {{
                    "success": true,
                    "critical_failure": false,
                    "dependency_setup": true,
                    "analysis": "Successfully installed all required packages",
                    "next_command": null,
                    "alternative_command": null
                }}
                {{
                    "success": false,
                    "critical_failure": false,
                    "dependency_setup": false,
                    "analysis": "The cd command failed because directory doesn't exist",
                    "next_command": null,
                    "alternative_command": "ls -la"
                }}
                {{
                    "success": false,
                    "critical_failure": false,
                    "dependency_setup": false,
                    "analysis": "Failed to install pandas package. This might be due to Python version incompatibility or missing system dependencies.",
                    "next_command": "python --version",
                    "alternative_command": "pip install pandas==1.2.0"
                }}
            """
        }
        
        # Call LLM with the user message
        response = call_llm([user_message], system_message=system_prompt, context=context)
        analysis = json.loads(response)
        
        context.add_chat_message('assistant', response)
        
        analysis = json.loads(response)
        
        return (
            analysis["success"], 
            log_output, 
            analysis.get("next_command"),
            analysis.get("critical_failure", False),
            analysis.get("alternative_command"),
            analysis.get("dependency_setup", False)
        )
    except Exception as e:
        return (False, f"Error executing command: {str(e)}\n{log_output}", None, True, None, False)

def run_from_github(github_url, README):
    """
    Clones a GitHub repository, reads the README file, extracts command-line instructions,
    executes them live, and returns the full log.
    """
    MAX_TOTAL_COMMANDS = 50  # Maximum total commands to execute
    MAX_RETRIES_PER_COMMAND = 3  # Maximum retries for each command
    command_count = 0
    retry_counts = {}  # Track retries for each command
    last_executable = None  # Track the last executable command
    
    # Add command execution history tracking
    command_history = []
    
    # Create command_history directory if it doesn't exist
    os.makedirs("command_history", exist_ok=True)
    
    log_history = ""  # Initialize log accumulator
    repo_dir = None 
    verification_done = False
    # Set up context
    context = RepositoryContext(github_url)
    
    # Step 1: Clone the repository
    try:
        log_history += f"Cloning repository: {github_url}\n"
        yield log_history  # Update UI
        repo_dir = clone_repo(github_url)
        
        # Record clone command
        command_history.append({
            "command": f"git clone {github_url} {repo_dir}",
            "output": f"Repository cloned to {repo_dir}"
        })
        
        log_history += f"Repository cloned to {repo_dir}\n"
        yield log_history
   
        # Step 2: Extract command-line instructions using GPT
        log_history += "Extracting command lines from README...\n"
        yield log_history
        commands = get_command_lines_from_readme(README, context=context)
        log_history += f"22222Extracted commands:\n{commands}\n"
        
        # Add chat history after command extraction
        log_history += "\n=== LLM Chat History ===\n"
        for msg in context.context['chat_history']:
            log_history += f"{msg['role'].upper()}: {msg['content']}\n"
        log_history += "=== End Chat History ===\n"
        yield log_history

        # Execute commands in a loop with GPT analysis
        command_list = commands.strip().split('\n')
        current_command = 0
        print("The command list is: ", command_list)
        
        while current_command < len(command_list):
            cmd = command_list[current_command].strip()
            if not cmd:
                current_command += 1
                continue
            
            # Check total command limit
            command_count += 1
            if command_count > MAX_TOTAL_COMMANDS:
                log_history += "\n⚠️ Maximum command execution limit reached. Stopping execution.\n"
                yield log_history
                break
                
            # Check retry limit for this specific command
            retry_counts[cmd] = retry_counts.get(cmd, 0) + 1
            if retry_counts[cmd] > MAX_RETRIES_PER_COMMAND:
                log_history += f"\n⚠️ Maximum retries reached for command: {cmd}. Skipping.\n"
                current_command += 1
                continue
                
            log_history += f"\n---\nExecuting command: {cmd} (Attempt {retry_counts[cmd]})\n"
            last_executable = cmd  # Update last executable command
            yield log_history
            
            success, output, next_cmd, critical_failure, alternative_cmd, dependency_set_up = execute_and_analyze_command(cmd, repo_dir, context)
            log_history += output
            
            # Record command execution
            command_history.append({
                "command": cmd,
                "output": output
            })
            
            # Check if dependencies are set up
            if dependency_set_up and not verification_done:
                log_history += "\n🔍 Starting repository exploration and verification...\n"
                yield log_history
                
                verified, verification_output = explore_and_verify_setup(repo_dir, context)
                verification_done = True
                
                if verified:
                    log_history += "\n✅ Final Verification: Dependencies are correctly set up and working.\n"+"$Analysis$: "+verification_output+"$/Analysis$"
                else:
                    log_history += "\n⚠️ Final Verification failed: Dependencies might not be fully functional.\n"+"$Analysis$: "+verification_output+"$/Analysis$"
                
                yield log_history
                break
            
            if not success:
                if alternative_cmd:
                    log_history += f"⚠️ Command failed. Trying alternative command: {alternative_cmd}\n"
                    command_list.insert(current_command + 1, alternative_cmd)
                elif critical_failure:
                    log_history += "❌ Critical failure. Stopping execution.\n"
                    yield log_history
                    break
                else:
                    log_history += "⚠️ Command failed but continuing execution.\n"
                
            if next_cmd:
                command_list.insert(current_command + 1, next_cmd)
                
            current_command += 1
            yield log_history
        
        # Store the chat history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"LLM_repo/chat_history/chat_history_{timestamp}.txt", "w") as f:
            f.write(json.dumps(context.context['chat_history']))
            
        # Store command execution history
        with open(f"LLM_repo/command_history/command_history_{timestamp}.json", "w") as f:
            json.dump(command_history, f, indent=4, ensure_ascii=False)
            
        yield log_history

    except Exception as e:
        log_history += f"❌ Error cloning repository: {str(e)}\n"
        yield log_history

    finally:
        
        # Add chat history after each command analysis
        log_history += "\n=== Command Analysis Chat History ===\n"
        recent_messages = context.context['chat_history'][-2:]
        for msg in recent_messages:
            log_history += f"{msg['role'].upper()}: {msg['content']}\n"
        log_history += "=== End Analysis History ===\n"
        
        # Add last executable command to log
        if last_executable:
            log_history += f"\n=== Last Executable Command ===\n{last_executable}\n=== End Last Command ===\n"
            
        if repo_dir:
            shutil.rmtree(repo_dir, ignore_errors=True)
            log_history += f"\n🧹 Cleaned up repository: {repo_dir}\n"
            yield log_history

def get_final_log(generator):
    """
    Consumes a generator and returns the final log history.
    This function manually calls next() until StopIteration is raised,
    then returns the value carried by StopIteration if available, otherwise the last yielded log.
    """
    final = ""
    while True:
        try:
            final = next(generator)
        except StopIteration as e:
            # If the generator returns a value, use it; otherwise use the last yielded value.
            if e.value is not None:
                final = e.value
            break
    return final

def process_single_repo(url, readme):
    """
    Runs run_from_github for a single repository and returns a tuple (url, result_dict).
    """
    print("Process single repo, url:", url)
    log = run_from_github(url, readme)
    final_log = get_final_log(log)
    success = "\n✅ Final Verification: Dependencies are correctly set up and working.\n" in final_log
    
    # Extract analysis from verification output
    analysis = ""
    for line in final_log.split('\n'):
        if "$Analysis$:" in line:
            analysis = line.split("$Analysis$:")[1].split("$/Analysis$")[0].strip()
            break
    
    # If no verification analysis found, try to get the last command analysis
    if not analysis:
        chat_history_start = final_log.find("=== Command Analysis Chat History ===")
        if chat_history_start != -1:
            chat_history = final_log[chat_history_start:]
            try:
                # Try to parse the last analysis JSON
                analysis_start = chat_history.find('{\n')
                analysis_end = chat_history.find('\n}', analysis_start) + 2
                if analysis_start != -1 and analysis_end != -1:
                    analysis_json = json.loads(chat_history[analysis_start:analysis_end])
                    analysis = analysis_json.get("analysis", "")
            except json.JSONDecodeError:
                print(f"Failed to parse analysis JSON for {url}")
    
    # Extract the last executable command from the log
    last_executable = None
    for line in final_log.split('\n'):
        if line.startswith("Executing command: "):
            last_executable = line.replace("Executing command: ", "").split(" (Attempt")[0].strip()
            
    return url, {
        "log": final_log,
        "success": success,
        "last_executable": last_executable,
        "analysis": analysis
    }

def process_repos(repo_dict, max_workers=4):
    """
    Accepts a dictionary in one of two formats:
      - {github_url: readme_string}
      - {github_url: [readme_string, ...]}
    
    For each GitHub URL, it concurrently runs run_from_github(github_url, readme) and collects the final log.
    It returns a dictionary where each key is the GitHub URL and each value is a dictionary containing:
       - log: The complete execution log
       - success: Whether the setup was verified successfully
       - last_executable: The last executable command that was run
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each repository.
        future_to_url = {}
        for url, value in repo_dict.items():
            readme = value[0] if isinstance(value, list) else value
            future = executor.submit(process_single_repo, url, readme)
            future_to_url[future] = url
            
        # Collect results as they complete.
        for future in as_completed(future_to_url):
            try:
                url, result = future.result()
                results[url] = result
            except Exception as e:
                # In case of error, store the error message in the results.
                url = future_to_url[future]
                results[url] = {
                    "log": f"Error processing repo: {str(e)}",
                    "success": False,
                    "last_executable": None,
                    "analysis": str(e)
                }
                
    return results


def save_results_to_file(results, filename):
    """
    Saves the results dictionary to a JSON file.
    
    Args:
        results (dict): Dictionary of results.
        filename (str): File path to save the results.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


with open('LLM_repo/repo_set/repos_data.json', 'r', encoding='utf-8') as f:
    repo_dict = json.load(f)
print("Repo set loaded:", repo_dict)
repo_dict = dict(list(repo_dict.items())[:40])
results = process_repos(repo_dict)
save_results_to_file(results, "LLM_repo/results_larger_set.json")