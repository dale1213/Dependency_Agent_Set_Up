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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API key and instantiate OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Ensure it is set in the .env file.")

client = OpenAI(api_key=api_key)

##############################
# Existing Functions (File-based)
##############################


def get_command_lines_from_readme(README):
    """Use GPT to extract only the command-line instructions from the README-like text."""
    extracted_text = README

    prompt = (
        "You are given the contents of a README file below. "
        "Please extract and print only the command-line instructions. "
        "Ignore all other text. Remove triple backticks, etc.\n\n"
        f"{extracted_text}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    extracted_commands = response.choices[0].message.content
    # Remove any leftover markdown formatting (e.g., triple backticks)
    cleaned_commands = extracted_commands.replace("```bash", "").replace("```", "").strip()
    return cleaned_commands

def run_commands_live(command_text):
    """
    A generator that executes shell commands dynamically and streams output live,
    while preserving the full log history.
    """
    commands = command_text.strip().splitlines()
    error_detected = False
    log_history = ""  # Accumulate logs

    for cmd in commands:
        if not cmd.strip():
            continue  # Skip empty lines

        log_history += f"\n---\nRunning command: {cmd}\n"
        yield log_history  # Yield the full log so far

        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Stream stdout line-by-line
            for line in iter(process.stdout.readline, ''):
                log_history += line
                yield log_history

            # Check stderr for errors
            stderr_output = process.stderr.read()
            if stderr_output:
                log_history += stderr_output
                error_detected = True
                log_history += "\n❌ Error detected. Stopping execution.\n"
                yield log_history
                break

            exit_code = process.wait()
            if exit_code != 0:
                error_detected = True
                log_history += f"\n❌ Error: Command failed with exit code {exit_code}. Stopping.\n"
                yield log_history
                break

        except Exception as e:
            error_detected = True
            log_history += f"\n❌ Exception: {str(e)}\n"
            yield log_history
            break

    if not error_detected:
        log_history += "\n✅ Successfully executed all command lines without any errors.\n"
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

    # Step 1: Clone the repository
    try:
        log_history += f"Cloning repository: {github_url}\n"
        yield log_history  # Update UI
        repo_dir = clone_repo(github_url)
        log_history += f"Repository cloned to {repo_dir}\n"
        yield log_history
    except Exception as e:
        log_history += f"❌ Error cloning repository: {str(e)}\n"
        yield log_history
        return

    # Step 2: Extract command-line instructions using GPT
    log_history += "Extracting command lines from README...\n"
    yield log_history
    commands = get_command_lines_from_text(README)
    log_history += f"Extracted commands:\n{commands}\n"
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

    # Cleanup the cloned repository
    shutil.rmtree(repo_dir)

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
    return url, [log, success]

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