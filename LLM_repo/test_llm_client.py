#!/usr/bin/env python
"""
Simple LLM Client Test
----------------------
A minimal script to test if the LLMClient is working correctly.
"""

import os
from dotenv import load_dotenv
from llm_call.LLMClient import LLMClient, call_llm

# Load environment variables
load_dotenv()

def main():
    """Run a simple test of the LLMClient."""
    
    # 1. Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or directly in the environment.")
        return
    
    print("Testing LLMClient with OpenAI...")
    print("-" * 50)
    
    # 2. Simple test with direct client creation
    try:
        # Create client
        client = LLMClient.create(provider="openai", model_name="gpt-3.5-turbo")
        
        # Test message
        messages = [
            {"role": "user", "content": "Hello! Please respond with a short message to confirm you're working."}
        ]
        
        # Make the call
        print("Sending request to OpenAI...")
        response = client.call(messages)
        
        # Print response
        print("\nResponse from OpenAI:")
        print(response)
        print("\nDirect client test: SUCCESS")
        
    except Exception as e:
        print(f"\nDirect client test FAILED: {str(e)}")
    
    print("-" * 50)
    
    # 3. Test with the wrapper function
    try:
        # Test message
        messages = [
            {"role": "user", "content": "Tell me the current date and a fun fact about today in history."}
        ]
        
        # System message
        system_message = "You are a helpful assistant. Please provide brief and accurate responses."
        
        # Make the call through the wrapper
        print("\nSending request through call_llm wrapper...")
        response = call_llm(
            messages=messages,
            system_message=system_message,
            model="gpt-3.5-turbo"
        )
        
        # Print response
        print("\nResponse through wrapper:")
        print(response)
        print("\nWrapper function test: SUCCESS")
        
    except Exception as e:
        print(f"\nWrapper function test FAILED: {str(e)}")
    
    print("-" * 50)
    print("All tests completed.")

if __name__ == "__main__":
    main() 