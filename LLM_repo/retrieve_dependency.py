import os
import re
import subprocess
from langchain_aws import ChatBedrock
from langchain.schema import AIMessage
from prompt_manager import LLMPromptManager

class PackageDependencyAnalyzer:
    """Analyzes a Python package to determine its build dependencies."""
    
    def __init__(self, dir):
        self.download_dir = f'{dir}/tmp_download/'
        os.makedirs(self.download_dir, exist_ok=True)
    
    def get_dep_setup(self, package_name, version):
        package_spec = f"{package_name}"
        if version:
            package_spec = f"{package_name}=={version}"
        
        cmd = [
            "pip", "download", 
            "--no-binary", ":all:", 
            "--no-deps",
            "-d", self.download_dir,
            package_spec
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        process.communicate()
        
        files = os.listdir(self.download_dir)
        if not files:
            return "No files were downloaded"
        
        package_pattern = re.compile(f"{re.escape(package_name)}.*\\.(tar\\.gz|zip|whl)")
        matching_files = [f for f in files if package_pattern.match(f)]
        
        if not matching_files:
            raise Exception(f"Downloaded file does not match expected pattern: {files}")
        
        package_path = os.path.join(self.download_dir, matching_files[0])

        if package_path.endswith('.tar.gz') or package_path.endswith('.tgz'):
            cmd = ["tar", "-xzf", package_path, "-C", self.download_dir]
        elif package_path.endswith('.zip'):
            cmd = ["unzip", package_path, "-d", self.download_dir]
        elif package_path.endswith('.whl'):
            cmd = ["unzip", package_path, "-d", self.download_dir]
        else:
            raise Exception(f"Unsupported archive format: {package_path}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Cannot unzip {package_path}")

        extracted_items = os.listdir(self.download_dir)
        package_dirs = [item for item in extracted_items if os.path.isdir(os.path.join(self.download_dir, item))]

        package_dir = os.path.join(self.download_dir, package_dirs[0])
        
        cmd = ["ls", "-la", package_dir]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = process.communicate()
        return stdout, package_dir

    def get_dep_info(self, package_name, package_dir, dep_bash):
        bash_script_path = f"{self.download_dir}/{package_name}-dep-retrieve.sh"
        script_header = f"#!/bin/bash\ncd {package_dir}\n\n"

        with open(bash_script_path, "w") as f:
            f.write(script_header + dep_bash)
        os.chmod(bash_script_path, 0o755)

        process = subprocess.Popen(
            ["bash", bash_script_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        output, _ = process.communicate()
        # TODO: handle error
        if process.returncode != 0:
            raise Exception("Dependency Retrieval Fail")
        return output
        

if __name__ == "__main__":
    package_name = 'tokenizers'
    version = '0.10.3'
    result = {
        "package": package_name,
        "version": version,
        "success": False,
        "steps": [],
        "build_analysis": None
    }
    dir = 'model_dep_resolution/papluca-xlm-roberta-base-language-detection'
    analyzer = PackageDependencyAnalyzer(dir)
    llm = ChatBedrock(region="us-east-1", model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    pm = LLMPromptManager(llm)
    dep_retrieval_pipeline = pm.create_dependency_retrieval_pipeline()
    dep_analysis_pipeline = pm.create_dependency_analysis_pipeline()
    
    pkg_ver = f"{package_name}=={version}"
    message, package_dir = analyzer.get_dep_setup(package_name, version)
    response = dep_retrieval_pipeline.invoke({
        "package_name": pkg_ver,
        "file_listing": message
    })

    if isinstance(response, AIMessage):
        dep_info_script = response.content
    else:
        dep_info_script = str(response)
    
    print(f'This is response: {dep_info_script}')

    dep_info = analyzer.get_dep_info(pkg_ver, package_dir, dep_info_script)
    analyzer_msg = dep_analysis_pipeline.invoke({
        "package_name": pkg_ver,
        "file_contents": dep_info
    })

    if isinstance(analyzer_msg, AIMessage):
        analyzer_response = analyzer_msg.content
    else:
        analyzer_response = str(analyzer_msg)

    print(f"This is analyzer_response:\n {analyzer_response}")