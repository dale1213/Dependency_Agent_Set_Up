import os
import re
import subprocess
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
        return stdout
        
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
    
    # Step 1: Download the package
    message = analyzer.get_dep_setup(package_name, version)
    print(message)