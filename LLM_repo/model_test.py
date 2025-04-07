import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import concurrent.futures
from langchain_aws import ChatBedrock
from langchain.schema import AIMessage
from llm_call.LLMClient import LLMClient
from prompt_manager import LLMPromptManager


dir = 'model_dep_resolution'

class DependencyResolver:
    def __init__(self):
        pass

class BashScriptFeedbackLoop:
    BASH_SCRIPT_PATH = 'setup.sh'
    TEST_SCRIPT_PATH = 'tester.py'

    def __init__(
        self,
        model_name,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        region: str = "us-east-1",
        max_retries: int = 3,
        log_level: int = logging.INFO,
        error_patterns: list = None
    ):
        self.max_retries = max_retries

        self.llm = ChatBedrock(region=region, model_id=model_id)

        self._setup_logger(log_level, model_name)

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

        pm = LLMPromptManager(self.llm)
        self.bash_fix_pipeline = pm.create_dep_fix_pipeline()
        self.err_identify_pipeline = pm.create_dep_fix_pipeline()

    def _setup_logger(self, log_level, model_name):
        self.logger = logging.getLogger(model_name)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Write to a model-specific log file
        log_file = os.path.join(f"{dir}/{model_name}", f"{model_name}.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _contains_error_pattern(self, output):
        """Check if output contains any of the defined error patterns."""
        for pattern in self.error_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False
    
    def initialize_dep_resolver(self, model_dir, model):
        model_url = model.get('card_url', 'Not specified')
        model_type = model.get('model_type', 'Not specified')
        pipeline_tag = model.get('pipeline_tag', 'Not specified')
        base_model = model.get('card_metadata', {}).get('base_model', 'Not specified')
        deps = model.get('dependencies', [])
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
            self.logger.info("Cannot infer test script")
            return False, []
        
        model_deps = []
        python_version = None
        if not isinstance(deps, list):
            return True, model_deps
        for dep in deps:
            if len(dep) == 2:
                library, version = dep
                if library == 'python':
                    python_version = version
                elif version is not None:
                    model_deps.append(f"pip install {library}=={version}")
                else:
                    model_deps.append(f"pip install {library}")

        script_content = [
            "#!/bin/bash",
            "set -e",
            f"VENV_NAME={os.path.basename(model_dir)}_env",
            f'cd {model_dir}',
            'pip install virtualenv',
        ]
        
        if python_version:
            script_content.append(f"virtualenv -p python{python_version} ${{VENV_NAME}}")
        else:
            script_content.append("virtualenv ${VENV_NAME}")
        
        script_content.extend([
            "source ${VENV_NAME}/bin/activate",
        ])
        try:
            with open(f'{model_dir}/{self.BASH_SCRIPT_PATH}', "w") as f:
                f.write("\n".join(script_content))

            with open(f'{model_dir}/{self.TEST_SCRIPT_PATH}', "w") as f:
                f.write(generated_code)

        except Exception as e:
                self.logger.error("Unexpected error during Initialization")
        return True, model_deps
    
    def run_dep_installation(self, model_dir, model):
        tester_is_generated, model_deps = self.initialize_dep_resolver(model_dir, model)
        if not tester_is_generated:
            return
        
        bash_script_path = f'{model_dir}/{self.BASH_SCRIPT_PATH}'
        for attempt in range(1, self.max_retries + 1):
            for model_dep in model_deps:
                try:
                    with open(bash_script_path, "a") as f:
                        f.write(f'\n{model_dep}')
                    
                    # For logging purpose
                    with open(bash_script_path, "r") as f:
                        bash_script_content = f.read()

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
                        response = self.bash_fix_pipeline.invoke({
                            "bash_script_content": bash_script_content,
                            "bash_cmd": model_dep,
                            "error_message": output
                        })

                        if isinstance(response, AIMessage):
                            updated_script = response.content
                        else:
                            updated_script = str(response)

                        with open(bash_script_path, "w") as f:
                            f.write(updated_script)
                        break
                    else:
                        self.logger.info(f"[{os.path.basename(bash_script_path)}] Bash script executed successfully")
                        # self.logger.info("Bash script executed successfully")
                        return  # Exit the loop if successful
                except Exception as e:
                    self.logger.error(f"Unexpected error during attempt {attempt}: {str(e)}")

        self.logger.info(f"[{os.path.basename(bash_script_path)}] Script still failing after {self.max_retries} attempts. Exiting.")
    
    def run_model_test(self, model_dir):
        self.logger.info(f"Running Python test script: {model_dir}/tester.py")
    
        try:
            with open(f'{model_dir}/setup.sh', 'r') as f:
                bash_content = f.read()

            with open(f'{model_dir}/tester.py', 'r') as f:
                python_content = f.read()
            
            exec_bash_script = f'{model_dir}/test.sh'
            with open(exec_bash_script, "w") as f:
                f.write(f'{bash_content}\npython tester.py')
            
            os.chmod(exec_bash_script, 0o755)

            process = subprocess.Popen(
                ["bash", exec_bash_script], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            output, _ = process.communicate()
            
            if process.returncode == 0:
                self.logger.info("Python test script executed successfully")
                return True, None
            else:
                self.logger.error(f"Python test script failed with return code {process.returncode}")
                self.logger.error(f"Output/Error: {output}")
                
                response = self.err_identify_pipeline.invoke({
                    "python_script_content": python_content,
                    "bash_script_content": bash_content,
                    "error_message": output
                })
                
                self.logger.info(f"Error message analysis: {response}")
                return False, response
                
        except Exception as e:
            self.logger.error(f"Error running test script: {str(e)}")
            return False, str(e)

def process_model(dir, model) -> None:
    model_name = model['model_id'].replace('/', '-')
    model_dir = f"{dir}/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    feedback_loop = BashScriptFeedbackLoop(
        model_name,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region="us-east-1",
        max_retries=3
    )

    feedback_loop.run_dep_installation(model_dir, model)
    feedback_loop.run_model_test(model_dir)
    
def main():
    os.makedirs(dir, exist_ok=True)
    json_file = "model_cards_text-classificatio.json"
    with open(json_file, 'r') as f:
        data = json.load(f)

    # max_workers = int(os.cpu_count() // 2)
    max_workers = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(process_model, dir, model): model
            for model in data[2:3]
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            future.result()
            # try:
            #     future.result()
            # except Exception as exc:
            #     print(f"Model {model['model_id']} generated an exception: {exc}")

    # clear cache
    try:
        subprocess.run(
            ["bash", "clear_cache.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    except Exception as e:
        print(f"Error during cache cleanup: {str(e)}")

if __name__ == '__main__':
    main()