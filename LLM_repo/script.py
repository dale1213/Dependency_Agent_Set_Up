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
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(
    level=logging.DEBUG,
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

##############################
# LLM helper functions
##############################

def call_llm(messages, model="gpt-3.5-turbo"):
    """
    Sends a request to the LLM and logs the input and output.
    """
    logger.info("Sending LLM request with the following messages:")
    logger.debug(json.dumps(messages, indent=2))
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    output = response.choices[0].message.content.strip()
    logger.info("Received LLM response:")
    logger.debug(output)
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

def get_command_lines_from_readme(README):
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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        try:
            extracted_chunk = call_llm(messages)  # Call the LLM to extract commands
            cleaned_chunk = extracted_chunk.replace("```bash", "").replace("```", "").strip()
            extracted_commands.append(cleaned_chunk)
        except Exception as e:
            logger.error(f"Error processing chunk {idx + 1}: {str(e)}")
            continue

    final_commands = "\n".join(extracted_commands).strip()
    logger.info("Completed extraction of command lines from provided text.")
    logger.debug(f"Extracted commands: {final_commands[:200]}{'...' if len(final_commands) > 200 else ''}")

    return final_commands

def run_commands_live(command_text, timeout=60):
    """
    Executes shell commands dynamically, ensuring processes terminate properly.
    """
    commands = command_text.strip().splitlines()
    log_history = ""
    
    for cmd in commands:
        if not cmd.strip():
            continue

        log_history += f"\n---\nRunning command: {cmd}\n"
        yield log_history  # Streaming output

        try:
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)  # Ensure it doesn’t hang
                log_history += stdout
                log_history += stderr

                if process.returncode != 0:
                    log_history += f"\n❌ Command failed with exit code {process.returncode}\n"
                    yield log_history
                    break
            except subprocess.TimeoutExpired:
                process.kill()  # Force stop if taking too long
                log_history += f"\n❌ Timeout: Command took longer than {timeout} seconds.\n"
                yield log_history
                break

        except Exception as e:
            log_history += f"\n❌ Exception while executing: {str(e)}\n"
            yield log_history
            break

    yield log_history

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


def run_commands_with_extraction(file):
    """
    1. Extract commands from the README via GPT
    2. Yield live output as commands run
    """
    commands = get_command_lines_from_readme(file)
    if (not commands or "No file uploaded" in commands or
        "not supported" in commands or "No readable text" in commands):
        logger.warning("No valid commands extracted from the file.")
        yield commands
        return

    yield from run_commands_live(commands)

##############################
# New Functions for GitHub URL-based Execution
##############################

def clone_repo(github_url):
    """
    Clones the GitHub repository into a temporary directory.
    Returns the path to the cloned repository.
    """
    temp_dir = tempfile.mkdtemp()
    cmd = f"git clone {github_url} {temp_dir}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        shutil.rmtree(temp_dir)
        raise Exception(f"Failed to clone repository: {result.stderr}")
    return temp_dir

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

def run_from_github(github_url, README):
    """
    Clones a GitHub repository, reads the README file, extracts command-line instructions,
    executes them live, and returns the full log.
    """
    log_history = ""  # Initialize log accumulator
    repo_dir = None 
    # Step 1: Clone the repository
    try:
        log_history += f"Cloning repository: {github_url}\n"
        yield log_history  # Update UI
        repo_dir = clone_repo(github_url)
        log_history += f"Repository cloned to {repo_dir}\n"
        yield log_history
   

        # Step 2: Extract command-line instructions using GPT
        log_history += "Extracting command lines from README...\n"
        yield log_history
        commands = get_command_lines_from_readme(README)
        log_history += f"22222Extracted commands:\n{commands}\n"
        yield log_history

        # Step 3: Execute the commands live and stream output
        log_history += "Executing extracted commands...\n"
        yield log_history
        for output in run_commands_live(commands):
            log_history += output  # Append output dynamically
            yield log_history  # Yield the full updated log

        # Step 4: Run 'ls -la' in the repository to capture its contents
        log_history += "Listing repository contents with 'ls -la'...\n"
        yield log_history
        ls_output = run_ls(repo_dir)
        log_history += f"ls output:\n{ls_output}\n"
        yield log_history

        # Step 5: Ask ChatGPT for the executable command(s) based on the ls output
        log_history += "Extracting executable command from ls output...\n"
        yield log_history
        executable_command = get_executable_command_from_ls(ls_output)
        log_history += f"Extracted executable command(s):\n{executable_command}\n"
        yield log_history

        if not executable_command:
            log_history += "❌ Could not determine an executable command from ls output.\n"
            yield log_history
            shutil.rmtree(repo_dir)
            return

        # Step 6: Run the extracted executable command(s) in the repository directory
        log_history += "Running the executable command(s)...\n"
        yield log_history
        for output in run_executable(executable_command, working_dir=repo_dir):
            yield output

        # Step 7: Final verification: Check whether the executable actually ran successfully
        if "✅ Executable command ran successfully." in log_history:
            log_history += "\n✅ Final Verification: The executable command ran successfully.\n"
        else:
            log_history += "\n❌ Final Verification: The executable command did not run successfully.\n"
        yield log_history

    except Exception as e:
        log_history += f"❌ Error cloning repository: {str(e)}\n"
        yield log_history

    # Cleanup the cloned repository
    finally:
        # Ensure the repository is deleted even if an error occurs
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
    Runs run_from_github for a single repository and returns a tuple (url, (final_log, success_bool)).
    """
    log = run_from_github(url, readme)
    final_log = get_final_log(log)
    success = "\n✅ Final Verification: The executable command ran successfully.\n" in final_log
    return url, [final_log, success]

def process_repos(repo_dict, max_workers=4):
    """
    Accepts a dictionary in one of two formats:
      - {github_url: readme_string}
      - {github_url: [readme_string, ...]}
    
    For each GitHub URL, it concurrently runs run_from_github(github_url, readme) and collects the final log.
    It returns a dictionary where each key is the GitHub URL and each value is a tuple:
       (final_log, True)  if the success string is found,
       (final_log, False) otherwise.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each repository.
        future_to_url = {}
        for url, value in repo_dict.items():
            readme = value[0] if isinstance(value, list) else value
            future = executor.submit(process_single_repo, url, readme)
            future_to_url[future] = url
        executor.shutdown(wait=True) 
        # Collect results as they complete.
        for future in as_completed(future_to_url):
            try:
                url, result = future.result()
                results[url] = result
            except Exception as e:
                # In case of error, store the error message in the results.
                url = future_to_url[future]
                results[url] = (f"Error processing repo: {str(e)}", False)
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


if __name__ == "__main__":
    with open('repo_set/norepos.json', 'r', encoding='utf-8') as f:
        repo_dict = json.load(f)
    repo_dict = dict(list(repo_dict.items())[:10])
    results = process_repos(repo_dict)
    save_results_to_file(results, "results.json")