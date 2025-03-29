import os
import json
import logging
from llm_call.LLMClient import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_model_test_script(model, dir):
    model_url = model.get('card_url', 'Not specified')
    model_type = model.get('model_type', 'Not specified')
    pipeline_tag = model.get('pipeline_tag', 'Not specified')
    base_model = model.get('card_metadata', {}).get('base_model', 'Not specified')
    prompt = f'''
Given the model config in huggingface:
card_url: {model_url}, model_type: {model_type}, "pipeline_tag": {pipeline_tag}, "base_model": {base_model}
Based on those, please come up with a min-executable to test the model is compatible with the installed dependencies. 

Example code of the output:
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("some/unclear-model-name")
model = AutoModel.from_pretrained("some/unclear-model-name")

Please make sure your output code does not contain any comment or prints. 
If the information is not enough, return EXACTLY "Not enough information".
'''
    
    messages = [{"role": "user", "content": prompt}]
    try:
        client = LLMClient.create(provider="bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
        generated_code = client.call(messages, system_message=None, temperature=0)
    except Exception as e:
        print("Error calling LLMClient API:", e)
        raise SystemExit(1)
    if generated_code == 'Not enough information':
        return None
    else:
        filename = "tester.py"
        os.makedirs(dir, exist_ok=True)
        with open(f'{dir}/{filename}', "w") as f:
            f.write(generated_code)
        return f'{dir}/{filename}'


def get_model_info(model_data):
    model_id = model_data['model_id']
    deps = model_data.get('dependencies', [])
    model_deps = []
    for dep in deps:
        if len(dep) == 2:
            library, version = dep
            model_deps.append((library, version))
    
    return model_id, model_deps


def generate_bash_script(model, dir):
    model_id, model_deps = get_model_info(model)
    
    python_version = None
    filtered_deps = []
    for dep_name, dep_version in model_deps:
        if dep_name == 'python':
            python_version = dep_version
        else:
            filtered_deps.append((dep_name, dep_version))

    script_content = [
        "#!/bin/bash",
        "set -e",
        f'VENV_NAME="{dir}_env"',
        f'cd {dir}',
        'pip install virtualenv',
        ""
    ]
    
    if python_version:
        script_content.append(f"virtualenv -p python{python_version} ${{VENV_NAME}}")
    else:
        script_content.append("virtualenv ${VENV_NAME}")
    
    script_content.extend([
        "source ${VENV_NAME}/bin/activate",
    ])
    
    # Add dependencies
    for dep_name, dep_version in filtered_deps:
        if dep_version is not None:
            script_content.append(f"pip install {dep_name}=={dep_version}")
        else:
            script_content.append(f"pip install {dep_name}")
    script_content.append(f"python tester.py")
    bash_script = "\n".join(script_content)
    
    filename = "setup.sh"

    os.makedirs(dir, exist_ok=True)
    with open(f'{dir}/{filename}', "w") as f:
        f.write(bash_script)
    
    # Make the script executable
    os.chmod(f'{dir}/{filename}', 0o755)
    return f'{dir}/{filename}'

if __name__ == '__main__':
    json_file = "./model_analysis_type_question-answering_min_dl_1000_lib_transformers.json"
    with open(json_file, 'r') as f:
        data = json.load(f)

    for model in data:
        model_id = model['model_id']
        model_dir = model_id.replace('/', '_')
        get_model_test_script(model, model_dir)
        bash_name = generate_bash_script(model, model_dir)
        logger.info(f'Bash Script: {bash_name}')
        break