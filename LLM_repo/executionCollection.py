import os
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ExecutionAnalysisAgent:
    def __init__(self, openai_client):
        self.client = openai_client
        self.statistics = {
            'total_repos': 0,
            'success_count': 0,
            'failure_count': 0,
            'cluster_stats': {},  # Will store {category:issue: count}
            'issue_repos': {},    # Will store {category:issue: set(repo_urls)}
            'identified_patterns': set()  # Will store tuples of (pattern, repo_url)
        }
        
        # Reference clusters (for LLM context)
        self.cluster_categories = {
            'ENVIRONMENT': [
                'Python interpreter issues',
                'Permission and sudo issues',
                'Path and file location issues',
                'System configuration issues'
            ],
            'DEPENDENCY': [
                'Missing package issues',
                'Version compatibility issues',
                'Installation failures',
                'Dependency conflict issues'
            ],
            'COMMAND': [
                'Syntax errors',
                'Invalid command issues',
                'Timeout issues',
                'Python code in shell context',
                'Command sequence issues'
            ],
            'RESOURCE': [
                'Disk space issues',
                'Memory-related issues',
                'Network connectivity issues',
                'System resource limitations'
            ],
            'AGENT': [
                'Prompt engineering issues',
                'LLM response format errors',
                'Expected behavior misinterpretation',
                'Agent logic errors',
                'Natural language response instead of structured data'
            ]
        }

    def analyze_single_result(self, repo_url: str, result: dict) -> dict:
        """
        Analyze a single repository execution result using LLM.
        """
        # Prepare context for LLM
        analysis_prompt = self._create_analysis_prompt(result)
        
        try:
            # Get LLM analysis
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            # Parse LLM response
            try:
                analysis = json.loads(response.choices[0].message.content)
                # Validate required fields
                required_fields = ['success', 'primary_category', 'specific_issue', 'identified_patterns', 'analysis', 'suggested_fixes', 'confidence_score']
                for field in required_fields:
                    if field not in analysis:
                        raise ValueError(f"Missing required field: {field}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing LLM response for {repo_url}: {str(e)}")
                print(f"Raw response: {response.choices[0].message.content}")
                analysis = {
                    'success': result['success'],
                    'primary_category': 'AGENT',
                    'specific_issue': 'LLM response format errors',
                    'identified_patterns': [{'pattern': str(e), 'context': 'LLM response parsing error'}],
                    'analysis': f"Failed to parse LLM response: {str(e)}",
                    'suggested_fixes': [],
                    'confidence_score': 0.0
                }
            
            # Update statistics
            self._update_statistics(result['success'], analysis, repo_url)
            
            return {
                'repo_url': repo_url,
                'analysis': analysis,
                'raw_result': result
            }
            
        except Exception as e:
            print(f"Error analyzing {repo_url}: {str(e)}")
            # Create a default analysis for error cases
            error_analysis = {
                'success': result['success'],
                'primary_category': 'AGENT',
                'specific_issue': 'Agent logic errors',
                'identified_patterns': [{'pattern': str(e), 'context': 'Error during analysis'}],
                'analysis': f"Error during analysis: {str(e)}",
                'suggested_fixes': [],
                'confidence_score': 0.0
            }
            
            # Update statistics even for error cases
            self._update_statistics(result['success'], error_analysis, repo_url)
            
            return {
                'repo_url': repo_url,
                'analysis': error_analysis,
                'error': str(e),
                'raw_result': result
            }

    def _create_analysis_prompt(self, result: dict) -> str:
        """
        Create a detailed prompt for LLM analysis.
        """
        return f"""
            Please analyze this repository execution result and categorize any issues found.

            Execution Log:
            {result['log']}

            Last Executable Command:
            {result['last_executable']}

            Analysis from execution:
            {result['analysis']}

            Execution Status: {'Success' if result['success'] else 'Failure'}

            Please analyze this result and return a JSON response with the following structure:
            {{
                "success": {str(result['success']).lower()},
                "primary_category": string (one of: "ENVIRONMENT", "DEPENDENCY", "COMMAND", "RESOURCE"),
                "specific_issue": string (detailed subcategory),
                "identified_patterns": [
                    {{
                        "pattern": string (the error pattern identified),
                        "context": string (relevant log context)
                    }}
                ],
                "analysis": string (detailed explanation of what happened during execution),
                "suggested_fixes": [
                    string (potential solutions or improvements)
                ],
                "confidence_score": float (0-1)
            }}
            """

    def _get_system_prompt(self) -> str:
        """
        Create system prompt with cluster categories context.
        """
        return f"""You are an expert system analyzing repository execution results.
            Your task is to categorize execution results into appropriate categories while identifying specific issues and patterns.

            Reference Categories:
            {json.dumps(self.cluster_categories, indent=2)}

            Guidelines:
            1. Always provide analysis in the requested JSON format
            2. Be specific about error patterns identified
            3. Consider multiple potential issues in the same execution
            4. Provide confidence score based on clarity of error patterns
            5. Suggest specific, actionable fixes
            6. Look for both explicit and implicit error indicators
            """

    def _update_statistics(self, success: bool, analysis: dict, repo_url: str) -> None:
        """
        Update agent statistics based on analysis result.
        """
        self.statistics['total_repos'] += 1
        
        if success:
            self.statistics['success_count'] += 1
        else:
            self.statistics['failure_count'] += 1
            
        # Update cluster stats and repo sets
        category = analysis['primary_category']
        issue = analysis['specific_issue']
        category_key = f"{category}:{issue}"
        
        # Update count
        self.statistics['cluster_stats'].setdefault(category_key, 0)
        self.statistics['cluster_stats'][category_key] += 1
        
        # Update repo set
        if category not in self.statistics['issue_repos']:
            self.statistics['issue_repos'][category] = {}
        if issue not in self.statistics['issue_repos'][category]:
            self.statistics['issue_repos'][category][issue] = set()
        self.statistics['issue_repos'][category][issue].add(repo_url)
        
        # Track new patterns with their repo URLs
        for pattern in analysis['identified_patterns']:
            self.statistics['identified_patterns'].add((pattern['pattern'], repo_url))

    def get_statistics(self) -> dict:
        """
        Get comprehensive analysis statistics.
        """
        total = self.statistics['total_repos']
        if total == 0:
            return {'error': 'No repositories analyzed'}
            
        return {
            'summary': {
                'total_repositories': total,
                'success_count': self.statistics['success_count'],
                'failure_count': self.statistics['failure_count'],
                'success_rate': self.statistics['success_count'] / total,
                'failure_rate': self.statistics['failure_count'] / total
            },
            'cluster_distribution': self.statistics['cluster_stats'],
            'issue_repositories': self.statistics['issue_repos'],
            'identified_patterns': list(self.statistics['identified_patterns']),  # Now contains (pattern, repo_url) tuples
            'cluster_categories': self.cluster_categories
        }

# Usage example
def analyze_results(results_file: str) -> dict:
    """
    Analyze all results from a results file.
    """
    # Initialize agent
    agent = ExecutionAnalysisAgent(client)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Analyze each result
    analyses = {}
    for repo_url, result in results.items():
        analysis = agent.analyze_single_result(repo_url, result)
        analyses[repo_url] = analysis
        
    # Get overall statistics
    statistics = agent.get_statistics()
    
    return {
        'analyses': analyses,
        'statistics': statistics
    }

def save_results_to_file(results: dict, filename: str) -> None:
    """Save analysis results to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, default=lambda x: list(x) if isinstance(x, set) else x)

if __name__ == "__main__":
    results_file = "LLM_repo/results_larger_set.json"
    analysis = analyze_results(results_file)
    save_results_to_file(analysis['statistics'], "LLM_repo/results_larger_set_statistics.json")