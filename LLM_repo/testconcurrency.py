import os
import subprocess
import venv
import shutil
import itertools
import tempfile
import concurrent.futures
from llm_call.LLMClient import LLMClient

# --- Sample Data ---
model_cards = [
    {
        "model_id": "openai-community/gpt2",
        "card_url": "https://huggingface.co/openai-community/gpt2",
        "dependencies": [
            ["transformers", None]  # not used; we'll use all dependencies in library_versions
        ]
    }
]

library_versions = {
    "transformers": [
        # For testing, a couple of versions.
        "4.50.1", "4.3.0"
    ],
    "torch": [
        # For testing, a couple of versions.
        "2.5.1", "2.5.0"
    ],
    # Add additional dependencies here if needed.
}

def clean_generated_code(code: str) -> str:
    """
    Removes markdown code fences (lines starting with triple backticks)
    and extraneous preamble text so that only valid Python code remains.
    """
    lines = code.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
    for i, line in enumerate(cleaned_lines):
        stripped = line.strip()
        if (stripped.startswith("#!") or
            stripped.startswith("import ") or
            stripped.startswith("def ") or
            stripped.startswith("class ")):
            return "\n".join(cleaned_lines[i:])
    return "\n".join(cleaned_lines)

def generate_model_runner(card_url: str) -> str:
    """
    Uses LLMClient to generate a complete, standalone Python script that:
      - Downloads the GPT-2 model from Hugging Face using the transformers library.
      - Sets up a text-generation pipeline with a fixed seed (42).
      - Runs the pipeline on the prompt "Hello, I'm a language model," and prints the output.
    """
    prompt = (
        f"Given the Hugging Face model card URL: {card_url}, generate a complete, standalone Python script that does the following:\n\n"
        "- Downloads the GPT-2 model from Hugging Face using the transformers library.\n"
        "- Sets up a text-generation pipeline using the model.\n"
        "- Uses a fixed seed (e.g., 42) for reproducibility.\n"
        "- Runs the pipeline on the prompt \"Hello, I'm a language model,\" and prints the output.\n\n"
        "Provide only the code without any extra commentary."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        # Using the Bedrock provider with the specified model; adjust if needed.
        client = LLMClient.create(provider="bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
        generated_code = client.call(messages, system_message=None, temperature=0)
    except Exception as e:
        print("Error calling LLMClient API:", e)
        raise SystemExit(1)
    return clean_generated_code(generated_code)

def create_virtualenv(env_dir: str):
    """Creates a virtual environment in the specified directory."""
    print(f"Creating virtual environment at {env_dir} ...")
    venv.create(env_dir, with_pip=True)

def install_dependency(env_dir: str, package: str, version: str) -> bool:
    """Installs the specified package at the given version in the virtual environment."""
    python_path = os.path.join(env_dir, "bin", "python")
    # Create a custom pip cache directory inside the virtual environment.
    #custom_cache_dir = os.path.join(env_dir, ".pip_cache")
    #os.makedirs(custom_cache_dir, exist_ok=True)
    env_vars = os.environ.copy()
    #env_vars["PIP_CACHE_DIR"] = custom_cache_dir
    install_cmd = [
        python_path, "-m", "pip", "install", "--no-cache-dir",
        f"{package}=={version}"
    ]
    print("Installing dependency with command:", " ".join(install_cmd))
    result = subprocess.run(install_cmd, env=env_vars, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install {package}=={version}:\n{result.stderr}")
    return result.returncode == 0

def run_model(env_dir: str, runner_script: str) -> bool:
    """Runs the given runner script using the virtual environment's Python interpreter."""
    python_path = os.path.join(env_dir, "bin", "python")
    run_cmd = [python_path, runner_script]
    print("Running model with command:", " ".join(run_cmd))
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Run succeeded with output:\n", result.stdout)
    else:
        print("Run failed with error:\n", result.stderr)
    return result.returncode == 0

def test_combination(combo_dict: dict, runner_code: str):
    """
    Tests a single combination of dependency versions:
      - Creates a unique temporary directory for the virtual environment.
      - Installs each dependency at its specified version.
      - Writes the runner code into that environment.
      - Runs the runner script.
    Returns a tuple of the combination dictionary and a boolean success flag.
    """
    temp_dir = tempfile.mkdtemp(prefix="env_")
    print(f"Testing combination {combo_dict} in temporary directory {temp_dir}")
    try:
        create_virtualenv(temp_dir)
        for key, ver in combo_dict.items():
            print(f"Installing {key}=={ver} in {temp_dir}...")
            if not install_dependency(temp_dir, key, ver):
                print(f"Installation failed for {key}=={ver}.")
                shutil.rmtree(temp_dir)
                return (combo_dict, False)
        runner_script = os.path.join(temp_dir, "run_model.py")
        with open(runner_script, "w") as f:
            f.write(runner_code)
        os.chmod(runner_script, 0o755)
        success = run_model(temp_dir, runner_script)
    except Exception as e:
        print(f"Exception testing combination {combo_dict}: {e}")
        success = False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    return (combo_dict, success)

def main():
    # Generate the runner code once from the model card.
    card_url = model_cards[0]["card_url"]
    print("\n=== Processing model card:", card_url, "===")
    runner_code = generate_model_runner(card_url)
    
    # Prepare all combinations: the Cartesian product of dependency version lists.
    dep_keys = list(library_versions.keys())
    all_version_lists = [library_versions[key] for key in dep_keys]
    total_combinations = 1
    for lst in all_version_lists:
        total_combinations *= len(lst)
    print(f"Total combinations to test: {total_combinations}")
    
    combos = list(itertools.product(*all_version_lists))
    combos_dicts = [dict(zip(dep_keys, combo)) for combo in combos]
    
    successful_combinations = []
    max_workers = 2 #min(total_combinations, os.cpu_count() or 1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {executor.submit(test_combination, combo, runner_code): combo for combo in combos_dicts}
        for future in concurrent.futures.as_completed(future_to_combo):
            combo, success = future.result()
            if success:
                successful_combinations.append(combo)
    
    print("\n=== Summary of Successful Combinations ===")
    for comb in successful_combinations:
        print(comb)

if __name__ == "__main__":
    main()
