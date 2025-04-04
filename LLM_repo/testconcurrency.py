import os
import subprocess
import venv
import shutil
import itertools
import tempfile
import concurrent.futures
import json
from llm_call.LLMClient import LLMClient
import datetime

def load_model_cards_from_json(file_path: str) -> list:
    """
    Load model cards from a JSON file.
    Expected JSON format is a list of model objects:
    [
        {
            "model_id": "model/name",
            "card_url": "https://huggingface.co/model/name",
            "dependencies": [["package_name", "version"]]
        }
    ]
    """

    # --- Sample Data ---
    model_cards = [
        {
            "model_id": "bigwiz83/sapbert-from-pubmedbert-squad2",
            "card_url": "https://huggingface.co/bigwiz83/sapbert-from-pubmedbert-squad2",
            "dependencies": [
                ["transformers", None]  # not used; we'll use all dependencies in library_versions
            ]
        }
    ]

    if not os.path.exists(file_path):
        print(f"Warning: Model cards file {file_path} not found. Using default model cards.")
        return model_cards
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("Warning: JSON file does not contain a list of models. Using default model cards.")
                return model_cards
            return data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}. Using default model cards.")
        return model_cards
    except Exception as e:
        print(f"Unexpected error loading model cards: {e}. Using default model cards.")
        return model_cards


def get_library_versions(model_card: dict) -> dict:
    """
    Extracts library versions from model card dependencies.
    If a dependency doesn't specify a version, uses default versions.
    """

    # Default library versions if not specified in model card
    default_library_versions = {
        # "transformers": ["4.7.0"],
        # "torch": ["1.8.0"],
        # "datasets": ["1.4.1"],
        # "tokenizers": ["0.10.2"],
        # "huggingface-hub": ["0.0.8"],
        # "numpy": ["1.26.1"]
    }
    library_versions = default_library_versions.copy()
    
    # Update with versions from model card dependencies
    for dep in model_card.get("dependencies", []):
        print(f"Dependency: {dep}")
        package, version = dep
        library_versions[package] = [version]
    
    return library_versions

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
      - Downloads the specified model from Hugging Face
      - Sets up the appropriate pipeline based on the model type
      - Runs a simple test to verify the model works
    """
    prompt = (
        f"Given the Hugging Face model card URL: {card_url}, generate a complete, standalone Python script that:\n\n"
        "1. Imports necessary libraries (transformers, torch, etc.)\n"
        "2. Downloads the model from Hugging Face using the transformers library\n"
        "3. Sets up the appropriate pipeline based on the model type (e.g., text-generation, question-answering, etc.)\n"
        "4. Uses a fixed seed (42) for reproducibility\n"
        "5. Runs a simple test with appropriate input for the model type\n"
        "6. Prints the output\n\n"
        "Important requirements:\n"
        " - Make sure all code is valid Python syntax\n"
        " - Provide only the code without any extra commentary or markdown formatting.\n"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
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

def install_dependency(env_dir: str, package: str, version: str) -> tuple[bool, str]:
    """Installs the specified package at the given version in the virtual environment.
    Returns a tuple of (success, output) where output contains the installation output."""
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
    return result.returncode == 0, result.stdout + result.stderr

def store_error_classification(input_file: str, model_id: str, category: str, description: str, missing_deps: list[str]):
    """
    Store error classification information in a JSON file.
    The file will be named <input_file_name>_error_classification.json
    """
    # Create the output filename by replacing .json with _error_classification.json
    output_file = input_file.rsplit('.', 1)[0] + '_error_classification.json'
    
    try:
        # Load existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Initialize model entry if it doesn't exist
        if model_id not in data:
            data[model_id] = {
                "errors": [],
                "error_statistics": {
                    "total_errors": 0,
                    "categories": {},
                    "missing_dependencies": {}
                }
            }
        
        # Add new error entry
        error_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "category": category,
            "description": description,
            "missing_dependencies": missing_deps
        }
        data[model_id]["errors"].append(error_entry)
        
        # Update statistics
        data[model_id]["error_statistics"]["total_errors"] += 1
        data[model_id]["error_statistics"]["categories"][category] = \
            data[model_id]["error_statistics"]["categories"].get(category, 0) + 1
        
        # Update missing dependencies statistics
        for dep in missing_deps:
            data[model_id]["error_statistics"]["missing_dependencies"][dep] = \
                data[model_id]["error_statistics"]["missing_dependencies"].get(dep, 0) + 1
        
        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Error classification stored in {output_file}")
        
    except Exception as e:
        print(f"Error storing error classification: {e}")

def analyze_installation_output(output: str, model_id: str, input_file: str) -> tuple[list[str], str, str]:
    """
    Analyzes pip installation output using LLM to:
    1. Categorize the error type
    2. Identify missing dependencies
    Returns a tuple of (missing_deps, category, description)
    """
    prompt = (
        "Analyze the following error output and provide a JSON response with three fields:\n"
        "1. 'category': The type of error (one of: package_missing, version_conflict, installation_failure, "
        "model_inherent, missing_files, permission_error, memory_error, gpu_error, network_error, "
        "env_error, dependency_error, unknown_error)\n"
        "2. 'description': A brief description of the error\n"
        "3. 'missing_deps': An array of missing or incompatible dependencies. Each string should be either:\n"
        "   - A package name if it's missing\n"
        "   - A package name with version constraint if there's a version compatibility issue\n"
        "   - Empty array [] if no dependencies are missing\n\n"
        "Example response:\n"
        '{"category": "installation_failure", "description": "Package installation failed during compilation", "missing_deps": ["torch", "numpy<2.0.0"]}\n\n'
        "Error output to analyze:\n"
        f"{output}"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        client = LLMClient.create(provider="bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
        response = client.call(messages, system_message=None, temperature=0)
        analysis = json.loads(response)
        
        if not isinstance(analysis, dict) or 'category' not in analysis or 'description' not in analysis or 'missing_deps' not in analysis:
            print(f"Warning: LLM returned invalid response format: {response}")
            return [], "unknown_error", "Failed to analyze error output"
            
        print(f"\nError Category: {analysis['category']}")
        print(f"Description: {analysis['description']}")
        
        # Store the error classification
        store_error_classification(
            input_file,
            model_id,
            analysis['category'],
            analysis['description'],
            analysis['missing_deps']
        )
        
        return analysis['missing_deps'], analysis['category'], analysis['description']
    except Exception as e:
        print(f"Error analyzing installation output: {e}")
        return [], "unknown_error", "Failed to analyze error output"

def get_dependency_suggestion(missing_dep: str, model_id: str) -> tuple[str, str]:
    """Uses LLM to suggest a dependency version for the missing package."""
    prompt = (
        f"For the model {model_id}, the package {missing_dep} is missing. "
        "Please suggest a specific, stable version of this package.\n\n"
        "Requirements:\n"
        "1. Choose a stable release version (not alpha, beta, or rc versions)\n"
        "2. The version should be recent but not too new\n"
        "3. The version should be compatible with other common ML libraries\n"
        "4. Return ONLY a JSON object with exactly these fields:\n"
        "   {\n"
        '     "package": "package_name",\n'
        '     "version": "x.y.z"\n'
        "   }\n"
        "5. Do not include any other text or version lists in the response\n"
        "6. Use semantic versioning format (e.g., '2.11.1')\n\n"
        "Example response:\n"
        '{"package": "transformers", "version": "4.36.2"}'
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        client = LLMClient.create(provider="bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
        response = client.call(messages, system_message=None, temperature=0)
        suggestion = json.loads(response)
        return suggestion['package'], suggestion['version']
    except Exception as e:
        print(f"Error getting dependency suggestion: {e}")
        return missing_dep, "latest"

def update_model_cards_json(file_path: str, model_id: str, new_dependency: tuple[str, str]):
    """Updates the model_cards.json file with new dependency information."""
    try:
        with open(file_path, 'r') as f:
            model_cards = json.load(f)
        print("Passed parameters", file_path, model_id, new_dependency)
        # Find the model and update its dependencies
        for model in model_cards:
            if model['model_id'] == model_id:
                # Check if dependency already exists
                dep_exists = False
                for dep in model['dependencies']:
                    if dep[0] == new_dependency[0]:
                        dep[1] = new_dependency[1]
                        dep_exists = True
                        break
                
                if not dep_exists:
                    model['dependencies'].append(list(new_dependency))
                    print(f"Added new dependency {new_dependency[0]}=={new_dependency[1]} to model {model_id}")
                break
        
        print("updated model cards", model_cards)
        # Save updated model cards back to file
        with open(file_path, 'w') as f:
            json.dump(model_cards, f, indent=2)
        print(f"Updated {file_path} with new dependency {new_dependency[0]}=={new_dependency[1]}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def run_model(env_dir: str, runner_script: str, model_cards: list, model_cards_file: str) -> bool:
    """Runs the given runner script using the virtual environment's Python interpreter."""
    python_path = os.path.join(env_dir, "bin", "python")
    run_cmd = [python_path, runner_script]
    print("Running model with command:", " ".join(run_cmd))
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Run succeeded with output:\n", result.stdout)
        return True
    else:
        print("Run failed with error:\n", result.stderr)
        # Analyze error for missing dependencies and error category
        missing_deps, category, description = analyze_installation_output(
            result.stderr, 
            model_cards[0]["model_id"],
            model_cards_file
        )
        if missing_deps:
            print(f"Found missing dependencies: {missing_deps}")
            # Get suggestions for each missing dependency
            for dep in missing_deps:
                suggested_package, suggested_version = get_dependency_suggestion(dep, model_cards[0]["model_id"])
                print(f"Suggested dependency: {suggested_package}=={suggested_version}")
                # Update model_cards.json with the new dependency
                update_model_cards_json(model_cards_file, model_cards[0]["model_id"], 
                                     (suggested_package, suggested_version))
        return False

def test_combination(combo_dict: dict, runner_code: str, model_cards: list, model_cards_file: str):
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
            # If version is None, use latest version
            if ver is None:
                ver = "latest"
            print(f"Installing {key}=={ver} in {temp_dir}...")
            success, output = install_dependency(temp_dir, key, ver)
            if not success:
                # Analyze output for missing dependencies and error category
                missing_deps, category, description = analyze_installation_output(
                    output,
                    model_cards[0]["model_id"],
                    model_cards_file
                )
                if missing_deps:
                    print(f"Found missing dependencies: {missing_deps}")
                    # Get suggestions for each missing dependency
                    for dep in missing_deps:
                        suggested_package, suggested_version = get_dependency_suggestion(dep, model_cards[0]["model_id"])
                        print(f"Suggested dependency: {suggested_package}=={suggested_version}")
                        # Update model_cards.json with the new dependency
                        update_model_cards_json(model_cards_file, model_cards[0]["model_id"], 
                                             (suggested_package, suggested_version))
                print(f"Installation failed for {key}=={ver}.")
                shutil.rmtree(temp_dir)
                return (combo_dict, False)
        runner_script = os.path.join(temp_dir, "run_model.py")
        with open(runner_script, "w") as f:
            f.write(runner_code)
        os.chmod(runner_script, 0o755)
        success = run_model(temp_dir, runner_script, model_cards, model_cards_file)
    except Exception as e:
        print(f"Exception testing combination {combo_dict}: {e}")
        success = False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    return (combo_dict, success)

def load_combinations_from_json(file_path: str) -> dict:
    """Load existing combinations from JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_combinations_to_json(file_path: str, model_id: str, successful_combinations: list):
    """Save successful combinations to JSON file, avoiding duplicates."""
    existing_data = load_combinations_from_json(file_path)
    
    # If model exists, add new combinations without duplicates
    if model_id in existing_data:
        existing_combinations = existing_data[model_id]
        for combo in successful_combinations:
            if combo not in existing_combinations:
                existing_combinations.append(combo)
    else:
        # If model doesn't exist, create new entry
        existing_data[model_id] = successful_combinations
    
    # Save updated data back to file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

def process_single_model(model_card: dict, model_cards_file: str):
    """
    Process a single model card:
      - Get library versions
      - Generate runner code
      - Test dependency combinations
      - Save successful combinations
    """
    # Get library versions from the model card
    library_versions = get_library_versions(model_card)
    
    # Generate the runner code from the model card
    card_url = model_card["card_url"]
    model_id = model_card["model_id"]
    print("\n=== Processing model card:", card_url, "===")
    runner_code = generate_model_runner(card_url)
    
    # Prepare all combinations: the Cartesian product of dependency version lists
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
        future_to_combo = {executor.submit(test_combination, combo, runner_code, [model_card], model_cards_file): combo for combo in combos_dicts}
        for future in concurrent.futures.as_completed(future_to_combo):
            combo, success = future.result()
            if success:
                successful_combinations.append(combo)
    
    print("\n=== Summary of Successful Combinations ===")
    for comb in successful_combinations:
        print(comb)
    
    # Save successful combinations to JSON file
    json_file_path = "successful_combinations.json"
    save_combinations_to_json(json_file_path, model_id, successful_combinations)
    print(f"\nSaved successful combinations to {json_file_path}")

def main():
    """Main function that processes all model cards."""
    # Load model cards from JSON file
    model_cards_file = "model_cards_text-classificatio.json"
    model_cards = load_model_cards_from_json(model_cards_file)
    
    print(f"\nFound {len(model_cards)} model cards to process")
    
    # Process each model card
    for i, model_card in enumerate(model_cards, 1):
        print(f"\n=== Processing model card {i}/{len(model_cards)} ===")
        try:
            process_single_model(model_card, model_cards_file)
        except Exception as e:
            print(f"Error processing model card {model_card['model_id']}: {e}")
            continue
    
    print("\n=== All model cards processed ===")

if __name__ == "__main__":
    main()
