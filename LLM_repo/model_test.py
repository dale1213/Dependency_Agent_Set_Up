import os
import re
import json
import logging
import subprocess
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_aws import ChatBedrock
from langchain.schema import AIMessage
from langchain.schema.runnable import RunnableSequence
from llm_call.LLMClient import LLMClient

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


class BashScriptFeedbackLoop:
    def __init__(
        self,
        llm=None,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        region: str = "us-east-1",
        max_retries: int = 3,
        system_prompt: str = None,
        human_prompt: str = None,
        log_level: int = logging.INFO,
        error_patterns: list = None,
    ):
        self.max_retries = max_retries

        if llm is None:
            self.llm = ChatBedrock(region=region, model_id=model_id)
        else:
            self.llm = llm

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Create handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Define common error patterns to look for in output
        self.error_patterns = error_patterns or [
            r"command not found",
            r"No such file or directory",
            r"ImportError:",
            r"ModuleNotFoundError:",
            r"Error:",
            r"Failed:",
            r"E: Unable to locate package",
            r"Cannot find",
            r"Could not find",
            r"Permission denied",
        ]

        # Default system prompt if none provided
        self.system_prompt = system_prompt or (
            "You are a helpful DevOps assistant. You receive a bash script that "
            "installs packages and runs a Python script. Your task is to fix "
            "environment-related errors by modifying the bash script. "
            "Return only the updated bash script content in your final answer."
        )

        # Default human prompt template if none provided
        self.human_prompt = human_prompt or (
            "The current bash script is:\n\n"
            "-----\n"
            "{bash_script_content}\n"
            "-----\n\n"
            "It produced this error:\n\n"
            "{error_message}\n\n"
            "Please update the bash script to fix the error. If dependencies are missing, "
            "add installation commands. Return only the updated bash script (no markdown)."
        )

        system_message = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(self.human_prompt)
        self.chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        self.pipeline = self.chat_prompt | self.llm

    def _contains_error_pattern(self, output):
        """Check if output contains any of the defined error patterns."""
        for pattern in self.error_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False

    def run_feedback_loop(self, bash_script_path: str) -> None:
        with open(bash_script_path, "r") as f:
            bash_script_content = f.read()

        for attempt in range(1, self.max_retries + 1):
            self.logger.info(f"Attempt {attempt}/{self.max_retries}")
            try:
                with open(bash_script_path, "w") as f:
                    f.write(bash_script_content)

                os.chmod(bash_script_path, 0o755)

                process = subprocess.Popen(
                    ["bash", bash_script_path], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True
                )
                output, _ = process.communicate()
                return_code = process.returncode
                if return_code != 0 or self._contains_error_pattern(output):
                    if return_code != 0:
                        self.logger.error(f"Script failed with return code {return_code}")
                    else:
                        self.logger.warning(f"Script succeeded with return code 0, but error patterns detected in output")
                    
                    self.logger.error(f"Output/Error: {output}")
                    
                    # Ask LLM to produce a new version of the bash script
                    response = self.pipeline.invoke({
                        "bash_script_content": bash_script_content,
                        "error_message": output
                    })

                    if isinstance(response, AIMessage):
                        updated_script = response.content
                    else:
                        updated_script = str(response)  # Fallback in case it's not an AIMessage

                    bash_script_content = updated_script
                else:
                    self.logger.info("Bash script executed successfully")
                    return  # Exit the loop if successful
            except Exception as e:
                self.logger.error(f"Unexpected error during attempt {attempt}: {str(e)}")

        print(f"\nScript still failing after {self.max_retries} attempts. Exiting.")
    

if __name__ == '__main__':

    # json_file = "./model_analysis_type_question-answering_min_dl_1000_lib_transformers.json"
    # with open(json_file, 'r') as f:
    #     data = json.load(f)

    # for model in data:
    #     model_id = model['model_id']
    #     model_dir = model_id.replace('/', '_')
    #     get_model_test_script(model, model_dir)
    #     bash_name = generate_bash_script(model, model_dir)
    #     self.logger.info(f'Bash Script: {bash_name}')
    #     break

    feedback_loop = BashScriptFeedbackLoop(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region="us-east-1",
        max_retries=3
    )

    feedback_loop.run_feedback_loop("wrong_script.sh")