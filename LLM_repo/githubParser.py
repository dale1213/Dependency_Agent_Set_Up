import requests
import base64
from typing import Dict, List, Tuple, Optional

class GithubRepoParser:
    def __init__(self, token: Optional[str] = None):
        """初始化解析器
        Args:
            token: GitHub Personal Access Token
        """
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
            
    def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """获取指定文件内容"""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                content = base64.b64decode(response.json()["content"]).decode()
                return content
            return None
        except Exception as e:
            print(f"Error getting {path}: {str(e)}")
            return None
            
    def parse_requirements(self, content: str) -> Tuple[str, int]:
        """解析 requirements.txt 内容"""
        if not content:
            return "", 0
        
        lines = [
            line.strip() 
            for line in content.split("\n") 
            if line.strip() and not line.startswith("#")
        ]
        return "\n".join(lines), len(lines)
        
    def process_repo(self, repo_url: str) -> Dict[str, List]:
        """处理单个仓库"""
        # 从URL提取owner和repo名
        parts = repo_url.split("/")
        owner, repo = parts[-2], parts[-1]
        
        # 获取README内容
        readme = self.get_file_content(owner, repo, "README.md") or ""
        
        # 获取requirements内容
        req_content = self.get_file_content(owner, repo, "requirements.txt")
        req_text, req_count = self.parse_requirements(req_content)
        
        return {
            repo_url: [
                readme,
                req_text,
                req_count
            ]
        }

def main():
    # 示例使用
    parser = GithubRepoParser(token="your_github_token")  # 建议使用token
    repos = [
        "https://github.com/cudbg/Kitana-e2e",
        "https://github.com/dale1213/2020-Coding-Challenge"
    ]
    
    result = {}
    for repo in repos:
        result.update(parser.process_repo(repo))
        
    return result

if __name__ == "__main__":
    result = main()
    print(result)