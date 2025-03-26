#!/usr/bin/env python
"""
LLM Client Sample Usage
-----------------------
This script demonstrates how to use the LLMClient class to interact with 
different language models including OpenAI and Amazon Bedrock.
"""

import os
import sys
from dotenv import load_dotenv
from context.RepositoryContext import RepositoryContext

# Import our custom LLMClient classes
from llm_call.LLMClient import LLMClient, call_llm

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if necessary environment variables are set."""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your API keys or set them manually.")
        return False
    
    # Check for AWS credentials if testing Bedrock
    # if len(sys.argv) > 1 and sys.argv[1] == "bedrock":
    #     aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    #     aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        
    #     if not aws_key or not aws_secret:
    #         print("ERROR: AWS credentials not found.")
    #         print("To test Bedrock, please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
    #         return False
    
    return True

def example_1_basic_openai():
    """Example 1: Basic OpenAI API call."""
    print("\n=== Example 1: Basic OpenAI API Call ===")
    
    # Create a client directly
    client = LLMClient.create(provider="openai", model_name="gpt-3.5-turbo")
    
    # Define a simple message
    messages = [
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]
    
    # Call the model
    print("Sending request to OpenAI...")
    response = client.call(messages)
    
    # Print the response
    print("\nResponse from OpenAI:")
    print("-" * 50)
    print(response)
    print("-" * 50)

def example_2_with_system_message():
    """Example 2: OpenAI API call with system message."""
    print("\n=== Example 2: OpenAI with System Message ===")
    
    # Define messages with a system message
    messages = [
        {"role": "user", "content": "Write a short poem about artificial intelligence."}
    ]
    
    system_message = "You are a poetic assistant that expresses complex technical concepts through beautiful verse."
    
    # Call using the convenience function
    print("Sending request with system message...")
    response = call_llm(
        messages=messages,
        system_message=system_message,
        model="gpt-3.5-turbo"
    )
    
    # Print the response
    print("\nResponse with system message:")
    print("-" * 50)
    print(response)
    print("-" * 50)

def example_3_with_context():
    """Example 3: Multiple exchanges with context."""
    print("\n=== Example 3: Conversation with Context ===")
    
    # Create a repository context to maintain chat history
    context = RepositoryContext("sample-repo")
    
    # First exchange
    messages = [
        {"role": "user", "content": "Hello! Can you tell me about yourself?"}
    ]
    
    print("User: Hello! Can you tell me about yourself?")
    response1 = call_llm(messages, model="gpt-3.5-turbo", context=context)
    print("\nAssistant:", response1)
    
    # Second exchange (using the same context)
    messages = [
        {"role": "user", "content": "What capabilities do you have that might help me with coding?"}
    ]
    
    print("\nUser: What capabilities do you have that might help me with coding?")
    response2 = call_llm(messages, model="gpt-3.5-turbo", context=context)
    print("\nAssistant:", response2)
    
    # Print full conversation history
    print("\nFull conversation history in context:")
    print("-" * 50)
    
    # Use context's existing message generation capability
    try:
        # Generate messages without adding a new message
        conversation_messages = context.generate_messages(message=None, system_message=None)
        for msg in conversation_messages:
            if msg.get('role') == 'system':
                continue  # Skip system messages for readability
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"{role.capitalize()}: {content}")
    except Exception as e:
        print(f"Could not access conversation history: {str(e)}")
        print("Note: The RepositoryContext implementation may vary. Check its API for accessing messages.")
    
    print("-" * 50)

def example_4_bedrock():
    """Example 4: Amazon Bedrock API call with Claude model."""
    print("\n=== Example 4: Amazon Bedrock with Claude ===")
    
    try:
        # Create a Bedrock client
        # Use Claude 3 Sonnet which requires the Messages API
        model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        print(f"Creating client for model: {model_name}")
        
        client = LLMClient.create(provider="bedrock", model_name=model_name)
        
        # Define a simple message
        messages = [
            {"role": "user", "content": "Explain the concept of machine learning to a 5-year-old."}
        ]
        
        # Call the model
        print(f"Sending request to Amazon Bedrock (Claude 3)...")
        response = client.call(messages)
        
        # Print the response
        print("\nResponse from Claude via Bedrock:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    
    except Exception as e:
        print(f"Error testing Bedrock: {str(e)}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        print("Make sure you have proper AWS credentials configured with Bedrock access.")
        print("Check that your boto3 version is at least 1.35.0 for Bedrock Messages API support.")
        print("\nAvailable models in your AWS account may vary. Common Claude model IDs include:")
        print("- anthropic.claude-3-sonnet-20240229-v1:0")
        print("- anthropic.claude-3-haiku-20240307-v1:0")
        print("- anthropic.claude-instant-v1")
        print("Run 'aws bedrock list-foundation-models' to see available models in your account")

def example_5_custom_parameters():
    """Example 5: Using custom parameters like temperature."""
    print("\n=== Example 5: Custom Parameters ===")
    
    # Create a client
    client = LLMClient.create(provider="openai", model_name="gpt-3.5-turbo")
    
    # Define messages
    messages = [
        {"role": "user", "content": "Generate three creative startup ideas."}
    ]
    
    # With high temperature (more creative)
    print("Generating with high temperature (1.0)...")
    response_creative = client.call(
        messages=messages,
        temperature=1.0,
        max_tokens=200
    )
    
    # With low temperature (more focused)
    print("\nGenerating with low temperature (0.2)...")
    response_focused = client.call(
        messages=messages,
        temperature=0.2,
        max_tokens=200
    )
    
    # Print both responses
    print("\nCreative response (temperature=1.0):")
    print("-" * 50)
    print(response_creative)
    print("-" * 50)
    
    print("\nFocused response (temperature=0.2):")
    print("-" * 50)
    print(response_focused)
    print("-" * 50)

def list_available_bedrock_models():
    """LIST ALL AVAILABLE BEDROCK MODELS"""
    print("\n=== LIST ALL AVAILABLE BEDROCK MODELS ===")
    
    try:
        import boto3
        import json
        
        print("Connecting to AWS Bedrock service...")     
        # 使用bedrock服务(不是bedrock-runtime)来获取模型列表
        client = boto3.client('bedrock')
        
        print("Getting available foundation models...")
        response = client.list_foundation_models()
        
        print("\nAvailable models:")
        print("-" * 50)
        
        # 按提供商组织模型  
        models_by_provider = {}
        
        for model in response.get('modelSummaries', []):
            provider = model.get('providerName', 'Unknown')
            model_id = model.get('modelId', 'Unknown')
            
            if provider not in models_by_provider:
                models_by_provider[provider] = []
                
            models_by_provider[provider].append(model_id)
        
        for provider, models in models_by_provider.items():
            print(f"\nProvider: {provider}")
            for model_id in models:
                print(f"  - {model_id}")
        
        print("-" * 50)
        
        if 'Anthropic' in models_by_provider:
            print("\nAnthropic Claude models:")
            for model_id in models_by_provider['Anthropic']:
                if 'claude' in model_id.lower():
                    print(f"  - {model_id}")
        
        return models_by_provider
        
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        print("Ensure your AWS credentials have permission to access Bedrock service")
        return None

def main():
    """运行所有示例"""
    check_environment()
    
    # 添加命令行参数支持
    import sys
    
    if len(sys.argv) > 1:
        # 检查命令行参数
        if sys.argv[1] == "list-models":
            # 只列出可用的Bedrock模型
            list_available_bedrock_models()
            return
        elif sys.argv[1] == "bedrock":
            # 只运行Bedrock示例
            example_4_bedrock()
            return
    
    # 运行OpenAI示例
    # example_1_basic_openai()
    # example_2_with_system_message()
    # example_3_with_context()
    # example_5_custom_parameters()
    
    # 询问是否运行Bedrock示例(可能需要额外设置)
    try:
        run_bedrock = input("\n是否也要测试Amazon Bedrock功能? (y/n): ").strip().lower()
        if run_bedrock == 'y':
            list_available_bedrock_models()  # 首先列出可用模型
            example_4_bedrock()
    except:
        # 处理输入错误的情况(比如在非交互式环境中)
        pass

if __name__ == "__main__":
    main() 