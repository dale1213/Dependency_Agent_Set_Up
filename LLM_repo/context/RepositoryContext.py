from datetime import datetime
import tiktoken

class RepositoryContext:
    def __init__(self, github_url, model="gpt-3.5-turbo", max_tokens=4096):
        """
        Initialize repository context
        
        Args:
            github_url: Repository URL
            model: LLM model name
            max_tokens: Maximum token limit
        """
        self.github_url = github_url
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
        
        self.context = {
            'repo_url': github_url,
            'readme': '',
            'system_message': {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            'chat_history': [],
            'error_logs': [],
            'execution_history': []
        }
    def _count_tokens(self, text: str) -> int:
        """Calculate number of tokens in text"""
        return len(self.encoding.encode(text))
    
    def _trim_chat_history(self):
        """
        Trim chat history to ensure it doesn't exceed token limit
        Keeps most recent messages and removes older ones
        """
        if not self.context['chat_history']:
            return
            
        total_tokens = self._count_tokens(self.context['readme'])
        preserved_messages = []
        
        # Iterate through messages from newest to oldest
        for message in reversed(self.context['chat_history']):
            message_tokens = self._count_tokens(message['content'])
            if total_tokens + message_tokens < self.max_tokens:
                preserved_messages.append(message)
                total_tokens += message_tokens
            else:
                break
                
        # Restore correct chronological order
        self.context['chat_history'] = list(reversed(preserved_messages))
    
    def generate_messages(self, message=None, system_message=None):
        """
        Generate message list for OpenAI API
        """
        if system_message:
            current_system = {
                'role': 'system',
                'content': system_message
            }
        else:
            current_system = self.context['system_message']
            
        messages = [current_system]
        
        if self.context['chat_history']:
            messages.extend(self.context['chat_history'])
            
        if message:
            messages.append(message)
                
        self.context['messages'] = messages
        
        if message:
            self.context['chat_history'].append({
                'role': message['role'],
                'content': message['content'],
                'timestamp': datetime.now().isoformat()
            })
                
        return messages

    def update_readme(self, readme_content):
        """
        Update repository README content
        Content will be truncated if it exceeds token limit
        """
        readme_tokens = self._count_tokens(readme_content)
        if readme_tokens > self.max_tokens // 2:  # Reserve half of tokens for chat history
            # Truncate README content
            while readme_tokens > self.max_tokens // 2:
                readme_content = readme_content[:int(len(readme_content) * 0.9)]  # Reduce by 10% each time
                readme_tokens = self._count_tokens(readme_content)
            readme_content += "\n... (content truncated due to length)"
            
        self.context['readme'] = readme_content
        self._trim_chat_history()  # Ensure total tokens stay within limit
    
    def add_chat_message(self, role: str, content: str):
        """
        Add new chat message, automatically cleaning up old messages if needed
        
        Args:
            role: Message sender role ('user' or 'assistant')
            content: Message content
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.context['chat_history'].append(message)
        self._trim_chat_history()
    
    def get_full_context(self):
        """Get complete context"""
        return self.context
    
    def get_token_count(self) -> int:
        """Get total token count of current context"""
        total = 0
        total += self._count_tokens(self.context['readme'])
        for message in self.context['chat_history']:
            total += self._count_tokens(message['content'])
        return total
    
    def add_error(self, error_message):
        self.context['error_logs'].append({
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })

    def update_system_message(self, content: str):
        """
        Update system message
        
        Args:
            content: New system message content
        """
        self.context['system_message'] = {
            'role': 'system',
            'content': content
        }

    def get_attempt_count(self):
        
        return len([x for x in self.context['execution_history'] 
                   if x['command'].startswith('Attempt')])