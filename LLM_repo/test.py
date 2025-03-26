import os
import subprocess
import venv
import shutil
import itertools
from llm_call.LLMClient import LLMClient

# --- Sample Data ---
# Model card: used to generate the runner script.
model_cards = [
    {
        "model_id": "openai-community/gpt2",
        "card_url": "https://huggingface.co/openai-community/gpt2",
        "dependencies": [
            ["transformers", None]  # (Not used now; we'll use all dependencies in library_versions)
        ]
    }
]

# Provided library_versions mapping for candidate versions.
library_versions = {
    "transformers": [
        # First 3.x release in our window, then 4.x series up to early 2025
        "3.0.0", "4.50.1"
    ],
    "torch": [
        # PyTorch releases from mid-2020 onward
        "2.5.1", "2.5.0"
    ],
    
}

def clean_generated_code(code: str) -> str:
    """
    Cleans the generated code by removing markdown code fences (lines starting with triple backticks)
    and any extraneous preamble text that doesn't appear to be valid Python code.
    """
    lines = code.splitlines()
    # Remove lines that are markdown code fences
    cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
    # Find the first line that looks like valid Python code.
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
        # Using the Bedrock provider with the specified model
        client = LLMClient.create(provider="bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
        # Call the model with temperature=0 for deterministic output.
        generated_code = client.call(messages, system_message=None, temperature=0)
    except Exception as e:
        print("Error calling LLMClient API:", e)
        raise SystemExit(1)
    cleaned_code = clean_generated_code(generated_code)
    return cleaned_code

def create_virtualenv(env_dir: str):
    """Creates a virtual environment in the specified directory."""
    print(f"Creating virtual environment at {env_dir} ...")
    venv.create(env_dir, with_pip=True)

def install_dependency(env_dir: str, package: str, version: str) -> bool:
    """Installs the specified package at the given version in the virtual environment."""
    python_path = os.path.join(env_dir, "bin", "python")
    install_cmd = [python_path, "-m", "pip", "install", f"{package}=={version}"]
    print("Installing dependency with command:", " ".join(install_cmd))
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install {package}=={version}:\n{result.stderr}")
    return result.returncode == 0

def run_model(env_dir: str, script_path: str) -> bool:
    """Runs the generated runner script using the virtual environment's Python interpreter."""
    python_path = os.path.join(env_dir, "bin", "python")
    run_cmd = [python_path, script_path]
    print("Running model with command:", " ".join(run_cmd))
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Run succeeded with output:\n", result.stdout)
    else:
        print("Run failed with error:\n", result.stderr)
    return result.returncode == 0

def main():
    # Generate the runner script once from the model card.
    card_url = model_cards[0]["card_url"]
    print("\n=== Processing model card:", card_url, "===")
    runner_code = generate_model_runner(card_url)
    runner_script = "run_model.py"
    with open(runner_script, "w") as f:
        f.write(runner_code)
    os.chmod(runner_script, 0o755)
    
    # Prepare to iterate over every combination of dependency versions.
    dep_keys = list(library_versions.keys())
    all_version_lists = [library_versions[key] for key in dep_keys]
    total_combinations = 1
    for lst in all_version_lists:
        total_combinations *= len(lst)
    print(f"Total combinations to test: {total_combinations}")
    
    successful_combinations = []
    combo_count = 0
    for combo in itertools.product(*all_version_lists):
        combo_count += 1
        combo_dict = dict(zip(dep_keys, combo))
        print(f"\n--- Testing combination {combo_count}/{total_combinations} ---")
        print("Combination:", combo_dict)
        # Create a unique virtual environment directory name.
        env_dir = "env_" + "_".join(f"{k}_{v}" for k, v in combo_dict.items())
        create_virtualenv(env_dir)
        
        # Install each dependency with its selected version.
        all_installed = True
        for key, ver in combo_dict.items():
            print(f"Installing {key}=={ver} in {env_dir}...")
            if not install_dependency(env_dir, key, ver):
                print(f"Installation failed for {key}=={ver}. Skipping this combination.")
                all_installed = False
                break
        
        if not all_installed:
            shutil.rmtree(env_dir)
            continue
        
        # Optionally, install additional packages if the runner code requires them.
        # For example:
        # if not install_dependency(env_dir, "torch", "1.13.0"):
        #     shutil.rmtree(env_dir)
        #     continue
        
        # Run the generated runner script.
        if run_model(env_dir, runner_script):
            print("Combination succeeded!")
            successful_combinations.append(combo_dict)
        else:
            print("Combination failed.")
        
        shutil.rmtree(env_dir)
    
    os.remove(runner_script)
    
    print("\n=== Summary of Successful Combinations ===")
    for comb in successful_combinations:
        print(comb)

if __name__ == "__main__":
    main()
