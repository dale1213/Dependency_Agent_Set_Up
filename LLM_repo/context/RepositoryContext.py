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
    
    def _truncate_message_content(self, message, max_length=1048000):
        """
        Truncate message content if it exceeds max length.
        """
        if len(message['content']) <= max_length:
            return message
        
        marker = "[Skipped due to length constraints]"
        marker_length = len(marker)
        keep_length = max_length - marker_length
        head_length = keep_length // 2
        tail_length = keep_length - head_length
        
        truncated_content = message['content'][:head_length] + marker + message['content'][-tail_length:]
        return {
            'role': message['role'],
            'content': truncated_content,
            'timestamp': message.get('timestamp')
        }

    def generate_messages(self, message=None, system_message=None, max_total_length=6000000):
        """
        Generate message list for OpenAI API, ensuring messages don't exceed length limits
        both individually and in total.
        
        Args:
            message: New message to add
            system_message: System message to use (if None, use context's system message)
            max_total_length: Maximum total length of all messages in characters
            
        Returns:
            List of messages suitable for API call
        """
        if system_message:
            current_system = {
                'role': 'system',
                'content': system_message
            }
        else:
            current_system = self.context['system_message']
            
        # Always include system message
        system_msg = self._truncate_message_content(current_system)
        total_length = len(system_msg['content'])
        messages = [system_msg]
        
        # If we have a new message, make sure it's included
        new_msg = None
        if message:
            new_msg = self._truncate_message_content(message)
            total_length += len(new_msg['content'])
        
        # Add history from newest to oldest until we hit the limit
        history_messages = []
        if self.context['chat_history']:
            for hist_msg in reversed(self.context['chat_history']):
                truncated_msg = self._truncate_message_content(hist_msg)
                msg_length = len(truncated_msg['content'])
                
                # If adding this message would exceed the limit, skip it
                if total_length + msg_length > max_total_length:
                    # Add a marker message to show history was truncated
                    if not history_messages:
                        history_messages.append({
                            'role': 'system',
                            'content': '[Earlier conversation history was omitted due to length constraints]'
                        })
                    break
                    
                history_messages.append(truncated_msg)
                total_length += msg_length
        
        # Add history in correct order (oldest first)
        messages.extend(reversed(history_messages))
        
        # Add the new message at the end if we have one
        if new_msg:
            messages.append(new_msg)
        
        # Store in context
        self.context['messages'] = messages
        
        # Add new message to history if needed
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