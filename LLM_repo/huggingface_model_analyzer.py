import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from huggingface_hub import HfApi, ModelCard, ModelInfo
import boto3
from botocore.exceptions import ClientError
from LLM_repo.llm_call.LLMClient import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HuggingFaceAnalyzer')

class HuggingFaceModelAnalyzer:
    def __init__(self, output_file: str = "model_analysis.json"):
        """
        Initialize the analyzer
        
        Args:
            output_file: Path to save the analysis results
        """
        self.hf_api = HfApi()
        self.output_file = output_file
        self.results = []
        
        # Initialize LLM client
        self.llm_client = LLMClient.create(provider="bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
        
    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        """
        Get the model card from the repository
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            Optional[ModelCard]: Model card object if successful, None otherwise
        """
        try:
            # Load model card
            card = ModelCard.load(model_id)
            return card
        except Exception as e:
            logger.warning(f"Could not get model card for {model_id}: {str(e)}")
            return None
        
    def extract_dependencies(self, model_card: str) -> Optional[List[Tuple[str, Optional[str]]]]:
        """
        Use Claude to extract dependencies from model card
        
        Args:
            model_card: Model card content
            
        Returns:
            Optional[List[Tuple[str, Optional[str]]]]: List of tuples containing (package_name, version)
                where version can be None if not specified. Returns None if no dependencies found.
        """
        prompt = f"""Please analyze the following model card and extract the Python package dependencies.
        Return ONLY a list of tuples in the format [(package_name, version), ...] where version should be None if not explicitly specified in the model card.
        Do not include any other text or explanation.
        
        For example:
        - If you find "protobuf==3.20.0", return ("protobuf", "3.20.0")
        - If you find "icetk" without version, return ("icetk", None)
        
        Model Card:
        {model_card}
        """
        
        try:
            messages = [{
                "role": "user",
                "content": prompt
            }]
            
            # Use LLMClient to make the API call
            response = self.llm_client.call(
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the response string into a list of tuples
            try:
                # Remove any whitespace and convert string representation to actual list
                response = response.strip()
                
                # Handle case when LLM returns "None"
                if response.lower() == "none":
                    return None
                    
                if response.startswith('[') and response.endswith(']'):
                    # Use eval to safely convert string representation to list
                    dependencies = eval(response)
                    # Validate the format
                    if not isinstance(dependencies, list):
                        raise ValueError("Response is not a list")
                    for dep in dependencies:
                        if not isinstance(dep, tuple) or len(dep) != 2:
                            raise ValueError("Invalid dependency format")
                    
                    # Return None if no dependencies found
                    return dependencies if dependencies else None
                else:
                    raise ValueError("Response is not a list")
            except Exception as e:
                logger.error(f"Error parsing dependencies: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error extracting dependencies: {str(e)}")
            return None
    
    def analyze_model(self, model_id: str) -> Optional[Dict]:
        """
        Analyze a single model
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Get model card
            card = self.get_model_card(model_id)
            if not card:
                return None
                
            # Get model info
            model_info = self.hf_api.model_info(model_id)
            
            # Get repository info
            repo_info = self.hf_api.repo_info(model_id, repo_type="model")
            
            # Extract dependencies using Claude
            dependencies = self.extract_dependencies(card.content)
            
            # Create analysis result
            result = {
                "model_id": model_id,
                "model_name": model_info.id,
                "author": model_info.author,
                "downloads": model_info.downloads,
                "downloads_all_time": model_info.downloads_all_time,
                "likes": model_info.likes,
                "tags": model_info.tags,
                "card_url": f"https://huggingface.co/{model_id}",
                "dependencies": dependencies,
                "last_modified": model_info.last_modified.isoformat() if model_info.last_modified else None,
                "created_at": model_info.created_at.isoformat() if model_info.created_at else None,
                "analysis_date": datetime.now().isoformat(),
                "repo_size": repo_info.size if hasattr(repo_info, 'size') else None,
                "repo_stars": repo_info.stars if hasattr(repo_info, 'stars') else None,
                "repo_visibility": repo_info.private,
                "model_type": model_info.config.get("model_type", "unknown") if model_info.config else "unknown",
                "pipeline_tag": model_info.pipeline_tag,
                "task_categories": model_info.task_categories if hasattr(model_info, 'task_categories') else None,
                "license": model_info.license if hasattr(model_info, 'license') else None,
                "card_metadata": card.data.to_dict() if card.data else {},
                "card_text": card.text,
                "card_content": card.content,
                "library_name": model_info.library_name,
                "transformers_info": model_info.transformers_info.__dict__ if model_info.transformers_info else None,
                "safetensors": model_info.safetensors.__dict__ if model_info.safetensors else None,
                "model_index": model_info.model_index,
                "trending_score": model_info.trending_score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing model {model_id}: {str(e)}")
            return None
    
    def analyze_models(self, model_ids: List[str]):
        """
        Analyze multiple models
        
        Args:
            model_ids: List of Hugging Face model IDs
        """
        for model_id in model_ids:
            logger.info(f"Analyzing model: {model_id}")
            result = self.analyze_model(model_id)
            if result:
                self.results.append(result)
                # Save intermediate results after each successful analysis
                self.save_results()
    
    def save_results(self):
        """Save analysis results to JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {self.output_file}")

class ModelFetcher:
    # Common model types/pipeline tags
    MODEL_TYPES = [
        # "text-generation",
        "text-classification",
        "token-classification",
        "question-answering",
        # "translation",
        # "summarization",
        # "automatic-speech-recognition",
        # "image-classification",
        # "object-detection",
        # "image-segmentation",
        # "text-to-image",
        # "image-to-text",
        # "zero-shot-classification",
        # "sentiment-analysis",
        # "feature-extraction",
        # "fill-mask",
        # "zero-shot-image-classification",
        # "image-to-image",
        # "text-to-speech",
        # "speech-to-text"
    ]
    
    def __init__(self):
        """Initialize the model fetcher"""
        self.hf_api = HfApi()
        
    def generate_output_filename(self, 
                               model_type: Optional[str] = None,
                               min_downloads: int = 1000,
                               library: Optional[str] = None,
                               task: Optional[str] = None,
                               author: Optional[str] = None) -> str:
        """
        Generate output filename based on parameters
        
        Args:
            model_type: Pipeline tag
            min_downloads: Minimum downloads threshold
            library: Library name
            task: Task name
            author: Author name
            
        Returns:
            str: Generated filename
        """
        parts = []
        
        if model_type:
            parts.append(f"type_{model_type}")
        if min_downloads > 0:
            parts.append(f"min_dl_{min_downloads}")
        if library:
            parts.append(f"lib_{library}")
        if task:
            parts.append(f"task_{task}")
        if author:
            parts.append(f"author_{author}")
            
        # If no parameters specified, use timestamp
        if not parts:
            parts.append(f"all_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        return f"model_analysis_{'_'.join(parts)}.json"
        
    def fetch_models(self, 
                    model_type: Optional[str] = None,
                    max_models: int = 100,
                    min_downloads: int = 1000,
                    sort_by: str = "downloads",
                    library: Optional[str] = None,
                    task: Optional[str] = None,
                    author: Optional[str] = None) -> List[str]:
        """
        Fetch model IDs based on criteria
        
        Args:
            model_type: Pipeline tag (e.g., "text-generation", "text-classification")
            max_models: Maximum number of models to return
            min_downloads: Minimum number of downloads for a model
            sort_by: Field to sort by ("downloads", "likes", "trending_score", "last_modified", "created_at")
            library: Library name (e.g., "transformers", "pytorch")
            task: Task name (e.g., "fill-mask", "automatic-speech-recognition")
            author: Author/organization name
            
        Returns:
            List[str]: List of model IDs
        """
        try:
            # Build filter list with simple keywords
            filters = []
            
            # Add library filter if specified
            if library:
                filters.append(library)
            
            # Add task filter if specified
            if task:
                filters.append(task)
            
            # Get models from Hugging Face
            models = self.hf_api.list_models(
                filter=filters if filters else None,
                author=author,
                pipeline_tag=model_type,  # Use pipeline_tag for model type
                sort=sort_by,
                direction=-1,  # Descending order
                limit=max_models,
                full=True  # Get full model info
            )
            
            # Filter and process results
            model_ids = []
            for model in models:
                # Skip if downloads are below threshold
                if model.downloads < min_downloads:
                    continue
                    
                # Skip if model is private
                if model.private:
                    continue
                    
                model_ids.append(model.id)
                
                # Break if we've reached max_models
                if len(model_ids) >= max_models:
                    break
            
            logger.info(f"Found {len(model_ids)} models matching criteria")
            return model_ids
            
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []

def main():
    # Example usage of ModelFetcher
    fetcher = ModelFetcher()
    
    # Example: Analyze different model types
    for model_type in ModelFetcher.MODEL_TYPES:
        logger.info(f"Fetching models for type: {model_type}")
        
        # Get models for this type
        model_ids = fetcher.fetch_models(
            model_type=model_type,
            max_models=50,
            min_downloads=1000,
            library="transformers"
        )
        
        if model_ids:
            # Generate output filename based on parameters
            output_file = fetcher.generate_output_filename(
                model_type=model_type,
                min_downloads=1000,
                library="transformers"
            )
            
            # Analyze fetched models
            analyzer = HuggingFaceModelAnalyzer(output_file=output_file)
            analyzer.analyze_models(model_ids)
            analyzer.save_results()
            
            logger.info(f"Analysis saved to {output_file}")
        else:
            logger.warning(f"No models found for type: {model_type}")

if __name__ == "__main__":
    main() 