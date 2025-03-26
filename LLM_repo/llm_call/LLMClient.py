import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import openai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LLMClient')

class LLMClient(ABC):
    """
    LLM client abstract base class that defines the common interface for all LLM clients.
    Supports different LLM backend services like OpenAI and Amazon Bedrock.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the LLM client
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 60
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize client resources and authentication"""
        pass
    
    @abstractmethod
    def call(self, 
            messages: List[Dict[str, str]], 
            system_message: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None) -> str:
        """
        Call the LLM model to generate a response
        
        Args:
            messages: List of messages, each containing 'role' and 'content' keys
            system_message: System message
            temperature: Temperature parameter, controls randomness of output
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Content of the model's response
        """
        pass
    
    def prepare_messages(self, messages: List[Dict[str, str]], system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare the message list to be sent to the LLM
        
        Args:
            messages: Original message list
            system_message: System message
            
        Returns:
            List[Dict[str, str]]: Processed message list
        """
        prepared_messages = messages.copy()
        
        # If a system message is provided, add it to the beginning of the message list
        if system_message:
            # Check if there's already a system message
            has_system_message = any(msg.get('role') == 'system' for msg in prepared_messages)
            
            if has_system_message:
                # Replace existing system message
                for msg in prepared_messages:
                    if msg.get('role') == 'system':
                        msg['content'] = system_message
                        break
            else:
                # Add new system message
                prepared_messages.insert(0, {'role': 'system', 'content': system_message})
        
        return prepared_messages
    
    def create_error_response(self, error_message: str) -> str:
        """
        Create a response for error situations
        
        Args:
            error_message: Error message
            
        Returns:
            str: Formatted error response
        """
        logger.error(f"LLM call failed: {error_message}")
        return f"ERROR: {error_message}"
    
    @classmethod
    def create(cls, provider: str = "openai", model_name: Optional[str] = None) -> 'LLMClient':
        """
        Factory method to create appropriate LLM client instance
        
        Args:
            provider: Provider name, "openai" or "bedrock"
            model_name: Model name, if None then use default model
            
        Returns:
            LLMClient: Appropriate LLM client instance
        """
        if provider.lower() == "openai":
            default_model = "gpt-3.5-turbo"
            return OpenAIClient(model_name or default_model)
        elif provider.lower() == "bedrock":
            default_model = "anthropic.claude-v2"
            return BedrockClient(model_name or default_model)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Please use 'openai' or 'bedrock'")


class OpenAIClient(LLMClient):
    """
    Client implementation for OpenAI API
    """
    
    def _initialize(self):
        """Initialize OpenAI client"""
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized, using model: {self.model_name}")
    
    def call(self, messages, system_message=None, temperature=0.7, max_tokens=None):
        """
        Call LLM model using OpenAI API
        
        Args:
            messages: Message list
            system_message: System message
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            
        Returns:
            str: Content of the model's response
        """
        prepared_messages = self.prepare_messages(messages, system_message)
        
        # Try to call the API, retrying up to the set number of times
        retries = 0
        while retries <= self.max_retries:
            try:
                # Build request parameters
                params = {
                    "model": self.model_name,
                    "messages": prepared_messages,
                    "temperature": temperature,
                }
                
                # Add optional parameters
                if max_tokens:
                    params["max_tokens"] = max_tokens
                
                # Send request
                logger.debug(f"Sending request to OpenAI: {params}")
                response = self.client.chat.completions.create(**params)
                
                # Extract and return response content
                return response.choices[0].message.content.strip()
            
            except openai.APIError as e:
                retries += 1
                if retries > self.max_retries:
                    return self.create_error_response(f"OpenAI API error: {str(e)}")
                
                logger.warning(f"OpenAI API error (retry {retries}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)
            
            except Exception as e:
                return self.create_error_response(f"Unexpected error: {str(e)}")


class BedrockClient(LLMClient):
    """
    Client implementation for Amazon Bedrock
    """
    
    def _initialize(self):
        """Initialize Amazon Bedrock client"""
        try:
            # Import necessary libraries
            import boto3
            import json
            
            # Use boto3's default credential chain which handles:
            # 1. Environment variables
            # 2. Shared credential file (~/.aws/credentials)
            # 3. EC2 IAM role or ECS task role
            # This is the best practice for EC2 instances
            
            # Get the region from environment or use default
            aws_region = os.getenv("AWS_REGION", "us-east-1")

            # Create the Bedrock client using default credential provider chain
            self.client = boto3.client('bedrock-runtime', region_name=aws_region)
            
            # Model mapping table, different model providers have different parameter structures
            self.model_family = self.get_model_family(self.model_name)
            self.api_type = self.get_api_type(self.model_name)
            logger.info(f"Amazon Bedrock client initialized, using model: {self.model_name} with API type: {self.api_type}")
        
        except ImportError:
            raise ImportError("Using Amazon Bedrock requires the boto3 library. Please run: pip install boto3")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Amazon Bedrock client: {str(e)}")
            
    def get_api_type(self, model_name):
        """
        Determine which API type to use based on the model name
        
        Args:
            model_name: Bedrock model name
            
        Returns:
            str: API type ('invoke_model' or 'converse')
        """
        model_name_lower = model_name.lower()
        
        # Claude 3 models use the Messages API (converse)
        if "claude-3" in model_name_lower:
            return "converse"
        
        # Legacy models use the InvokeModel API
        return "invoke_model"

    def get_model_family(self, model_name):
        """
        Determine the model family based on the model name
        
        Args:
            model_name: Bedrock model name
            
        Returns:
            str: Model family name
        """
        model_name_lower = model_name.lower()
        if "claude" in model_name_lower or "anthropic" in model_name_lower:
            return "anthropic"
        elif "titan" in model_name_lower or "amazon" in model_name_lower:
            return "amazon"
        elif "llama" in model_name_lower or "meta" in model_name_lower:
            return "meta"
        elif "cohere" in model_name_lower:
            return "cohere"
        else:
            logger.warning(f"Unknown model family: {model_name}, using default 'anthropic'")
            return "anthropic"  # Default to Anthropic family

    def format_messages_for_model(self, messages, system_message):
        """
        Format messages according to different models and API types
        
        Args:
            messages: Message list
            system_message: System message
            
        Returns:
            str or dict: Formatted messages
        """
        if self.model_family == "anthropic":
            if self.api_type == "converse":
                # Format for Claude 3 models using Messages API
                formatted_messages = []
                
                # Add user/assistant messages
                for message in messages:
                    role = message['role']
                    content = message['content']
                    
                    if role == 'system':
                        continue  # System message handled separately
                    
                    # Map OpenAI roles to Anthropic roles
                    if role in ('user', 'human'):
                        formatted_messages.append({
                            "role": "user",
                            "content": [{"text": content}]
                        })
                    elif role in ('assistant', 'ai'):
                        formatted_messages.append({
                            "role": "assistant", 
                            "content": [{"text": content}]
                        })
                
                # Return formatted messages with system message if provided
                result = {
                    "messages": formatted_messages
                }
                
                if system_message:
                    result["system"] = system_message
                    
                return result
            else:
                # Claude model message format (legacy API)
                formatted_prompt = ""
                
                # Add system message
                if system_message:
                    formatted_prompt += f"\n\nHuman: <system>{system_message}</system>\n\n"
                
                # Add conversation messages
                for message in messages:
                    role = message['role']
                    content = message['content']
                    
                    if role == 'system':
                        # System message already handled above
                        continue
                    elif role == 'user' or role == 'human':
                        formatted_prompt += f"Human: {content}\n\n"
                    elif role == 'assistant' or role == 'ai':
                        formatted_prompt += f"Assistant: {content}\n\n"
                
                # Add final prompt
                formatted_prompt += "Assistant: "
                return formatted_prompt
        
        elif self.model_family == "cohere":
            # Cohere model message format
            formatted_messages = []
            
            for message in messages:
                role = "USER" if message['role'] in ('user', 'human') else "CHATBOT"
                formatted_messages.append({
                    "role": role,
                    "message": message['content']
                })
            
            return formatted_messages
        
        else:
            # Default format, applicable to most models
            return self.prepare_messages(messages, system_message)
    
    def call(self, messages, system_message=None, temperature=0.7, max_tokens=None):
        """
        Call LLM model using Amazon Bedrock
        
        Args:
            messages: Message list
            system_message: System message
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            
        Returns:
            str: Content of the model's response
        """
        try:
            # Format messages
            formatted_messages = self.format_messages_for_model(messages, system_message)
            
            if self.api_type == "converse" and self.model_family == "anthropic":
                # Use Messages API for Claude 3 models
                logger.debug(f"Sending request to Amazon Bedrock using Messages API, model: {self.model_name}")
                
                # Set up parameters for converse method
                params = {
                    "modelId": self.model_name,
                    "messages": formatted_messages["messages"],
                    "inferenceConfig": {
                        "temperature": temperature,
                        "maxTokens": max_tokens or 2048
                    }
                }
                
                # Add system message if provided
                if system_message:
                    params["system"] = system_message
                    
                # Send the request using converse method
                response = self.client.converse(**params)
                
                # Extract response from Messages API response format
                if "output" in response and "message" in response["output"]:
                    message_content = response["output"]["message"]["content"]
                    if isinstance(message_content, list) and len(message_content) > 0:
                        return message_content[0].get("text", "")
                
                # Fallback for unexpected response format
                return str(response)
            
            else:
                # Use standard InvokeModel API for other models
                # Build request body according to different model families
                if self.model_family == "anthropic":
                    request_body = {
                        "prompt": formatted_messages,
                        "temperature": temperature,
                        "max_tokens_to_sample": max_tokens or 2048,
                        "anthropic_version": "bedrock-2023-05-31"
                    }
                elif self.model_family == "cohere":
                    request_body = {
                        "message": messages[-1]['content'],  # Last message
                        "chat_history": formatted_messages[:-1],  # Chat history
                        "temperature": temperature,
                        "max_tokens": max_tokens or 2048
                    }
                elif self.model_family == "meta":
                    request_body = {
                        "prompt": formatted_messages[-1]['content'],  # Last message
                        "temperature": temperature,
                        "max_gen_len": max_tokens or 2048
                    }
                elif self.model_family == "amazon":
                    request_body = {
                        "inputText": formatted_messages[-1]['content'],  # Last message
                        "textGenerationConfig": {
                            "temperature": temperature,
                            "maxTokenCount": max_tokens or 2048
                        }
                    }
                else:
                    # Generic format
                    request_body = {
                        "messages": formatted_messages,
                        "temperature": temperature
                    }
                    if max_tokens:
                        request_body["max_tokens"] = max_tokens
                
                # Send request to Bedrock
                logger.debug(f"Sending request to Amazon Bedrock using InvokeModel API, model: {self.model_name}")
                response = self.client.invoke_model(
                    modelId=self.model_name,
                    body=json.dumps(request_body)
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                
                # Extract response content according to different model families
                if self.model_family == "anthropic":
                    return response_body.get("completion", "").strip()
                elif self.model_family == "cohere":
                    return response_body.get("text", "").strip()
                elif self.model_family == "meta":
                    return response_body.get("generation", "").strip()
                elif self.model_family == "amazon":
                    return response_body.get("results", [{}])[0].get("outputText", "").strip()
                else:
                    # Generic response parsing
                    return str(response_body)
        
        except Exception as e:
            return self.create_error_response(f"Amazon Bedrock call failed: {str(e)}")


# Wrapper function compatible with existing code
def call_llm(messages, system_message=None, model="gpt-3.5-turbo", context=None):
    """
    LLM call wrapper function compatible with existing code
    
    Args:
        messages: Message list
        system_message: System message
        model: Model name
        context: Repository context object
        
    Returns:
        str: Content of the model's response
    """
    # Combine chat history with new messages if context exists
    if context:
        # Get the new user message if it exists
        new_message = next((msg for msg in messages if msg['role'] == 'user'), None)
        
        # Generate messages using context, including chat history and new message
        # Use a safe limit for total message length (leave room for response)
        MAX_MESSAGE_LENGTH = 1048000  # Same as in LoopPipeline.py
        max_total_length = MAX_MESSAGE_LENGTH * 0.9  # 90% of the limit to leave room for response
        
        messages = context.generate_messages(
            message=new_message, 
            system_message=system_message,
            max_total_length=max_total_length
        )
        # System message is already included in the generated messages
        system_message = None
    
    # Detect model provider
    if "claude" in model.lower() or model.startswith("anthropic."):
        provider = "bedrock"
    else:
        provider = "openai"
    
    # Create appropriate client
    client = LLMClient.create(provider=provider, model_name=model)
    
    # Call model
    output = client.call(messages, system_message)
    
    # Record chat history (if context object is provided)
    if context:
        context.add_chat_message('assistant', output)
    
    return output 