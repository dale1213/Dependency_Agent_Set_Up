from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class LLMPromptManager:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def create_dep_fix_pipeline(self):
        """Creates a pipeline for fixing dependency installation errors"""
        
        system_prompt = (
            "You are a helpful DevOps assistant. You receive a bash script that "
            "installs packages to run a Python script. Your task is to fix "
            "environment-related errors by modifying the bash script. "
            "Return only the updated bash script content in your final answer."
        )
        
        human_prompt = (
            "The current bash script is:\n"
            "-----\n"
            "{bash_script_content}\n"
            "-----\n"
            "The current command causing an error:\n"
            "{bash_cmd}\n"
            "-----\n"
            "It produced this error:\n"
            "{error_message}\n"
            "Please update the bash script to fix the error. If dependencies are missing, "
            "add installation commands. Return only the updated bash script (no markdown)."
        )
        
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        return chat_prompt | self.llm_client
    
    def create_error_identifier_pipeline(self):
        """Creates a pipeline for identifying the root cause of errors (Python code vs. dependencies)"""
        
        system_prompt = (
            "You are a diagnostic expert for Python development environments. Your task is to analyze "
            "errors and determine their root cause. When presented with a Python error and the "
            "environment setup scripts, you will determine if the error is due to dependencies or code issues. "
            "You must structure your response in a specific JSON format to facilitate automated processing."
        )
        
        human_prompt = (
            "The Python script that's generating errors is:\n"
            "-----\n"
            "{python_script_content}\n"
            "-----\n"
            "The current bash script for installing dependencies is:\n"
            "-----\n"
            "{bash_script_content}\n"
            "-----\n\n"
            "The error that occurred is:\n"
            "{error_message}\n"
            "Please analyze this error and determine its root cause. Your response must be in this exact format:\n"
            "```json\n"
            "{\n"
            "  \"root_cause\": \"DEPENDENCY\",  // Use ONLY \"DEPENDENCY\" or \"PYTHON_CODE\"\n"
            "  \"explanation\": \"brief explanation of the error\",\n"
            "  \"affected_components\": [\"component1\", \"component2\"],  // List of affected dependencies or code elements\n"
            "  \"suggested_fix\": \"description of how to fix the issue\"\n"
            "}\n"
            "```\n"
            "Do not include any text outside of this JSON structure."
        )
        
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        return chat_prompt | self.llm_client
    
    def create_python_error_fix_pipeline(self):
        """Creates a pipeline for analyzing Python errors and identifying dependency issues"""
        
        system_prompt = (
            "You are a helpful Python expert. You receive Python code that's generating errors "
            "along with the current dependency installation script. Your task is to identify "
            "missing or misconfigured dependencies that are causing the Python errors. "
            "Return only the updated bash script with the necessary dependency changes."
        )
        human_prompt = (
            "The Python script that's generating errors is:\n\n"
            "-----\n"
            "{python_script_content}\n"
            "-----\n\n"
            "The current bash script for installing dependencies is:\n\n"
            "-----\n"
            "{bash_script_content}\n"
            "-----\n\n"
            "The Python error that occurred is:\n\n"
            "{error_message}\n\n"
            "Please analyze this error and update the bash script to fix any missing or "
            "misconfigured dependencies. Return only the updated bash script (no markdown)."
        )
        
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        return chat_prompt | self.llm_client

    def create_dependency_retrieval_pipeline(self):
        system_prompt = (
            "You are an expert in Python package installation troubleshooting, specializing in resolving "
            "building wheel errors and dependency conflicts. When a package fails to install due to "
            "build errors, your primary goal is to identify the underlying cause."
            
            "A common pattern is that users encounter errors when pip tries to build wheels for packages "
            "with compiled extensions. Your task is to examine the package source and identify files that "
            "will reveal these requirements and potential conflict sources. "
            
            "Respond with shell commands to display the contents of the most relevant files for "
            "diagnosing build failures. Be comprehensive but prioritize files most likely to reveal "
            "the cause of wheel building failures."
        )

        human_prompt = (
            "I'm analyzing the Python package '{package_name}' and need to identify which files contain\n"
            "dependency information (Python dependencies, system libraries, compilers, etc. required for installation).\n\n"
            "Here's a listing of files in the extracted package directory:"
            "-----\n"
            "{file_listing}\n"
            "-----\n"
            "The current bash script for installing dependencies is:\n\n"
            "-----\n"
            "Please provide bash commands to display the contents of ONLY the files that are most likely to contain:\n"
            "1. Python package dependencies\n"
            "2. Build system requirements\n"
            "3. System dependencies (like compiler requirements)\n"
            "4. Rust/C++/other language dependencies if this package has compiled extensions\n"
            
            "Format your response as a series of bash commands that include both echo statements to identify the file\n"
            "and cat commands to show the contents. For example:\n"
            "echo \"\\n========== pyproject.toml ==========\"\n"
            "cat pyproject.toml  # Modern build system requirements\n"
            "echo \"\\n========== setup.py ==========\"\n"
            "cat setup.py  # Traditional setup file with dependencies\n"
            "Return only the bash commands (no markdown)."
        )
        
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        return chat_prompt | self.llm_client
    
    def create_dependency_analysis_pipeline(self):
        system_prompt = (
            "You are an expert in Python package installation troubleshooting. Your task is to analyze "
            "the contents of dependency files from a Python package and identify why it might fail to build wheels. "
            "Focus on Python version requirements, system dependencies, and compiler/build tool requirements."
        )

        human_prompt = (
            "I'm trying to install the Python package '{package_name}' and need to understand what "
            "dependencies and environment requirements are needed. I've examined the package's dependency "
            "files, and here are their contents:\n\n"
            "{file_contents}\n\n"
            
            "Based on these file contents, please analyze the package dependencies and requirements. "
            "Then, provide a diagnostic bash script that will examine my current environment and determine "
            "if it meets all the necessary requirements for building this package successfully.\n\n"
            
            "The script should check for compatibility with Python versions, required system libraries, "
            "compiler toolchains, and any other dependencies you identify. It should output clear "
            "diagnostic information about what's compatible and what's missing or incompatible.\n\n"
            
            "Return only the updated bash script (no markdown)."
        )
        
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        return chat_prompt | self.llm_client