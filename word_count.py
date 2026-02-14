import re

def count_words_in_tex(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract Main Body (between \begin{document} and \appendix)
    start_match = re.search(r'\\begin\{document\}', content)
    end_match = re.search(r'\\appendix', content)
    
    if not start_match:
        print("Error: Could not find \\begin{document}")
        return

    start_idx = start_match.end()
    end_idx = end_match.start() if end_match else len(content)
    
    body = content[start_idx:end_idx]

    # 2. Cleanup LaTeX Commands
    # Remove comments
    body = re.sub(r'%.*', '', body)
    
    # Remove \begin{...} and \end{...} tags (retain content for environments like abstract, but maybe exclude figures/tables?)
    # For simplicity, we just strip the commands themselves but keep the text inside braces for things like \textbf{}
    
    # Remove figures and tables content? 
    # Usually "Main text" includes captions but not the raw float code. 
    # Let's just strip standard commands.
    
    # Remove TikZ pictures entirely as they are code
    body = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', body, flags=re.DOTALL)
    
    # Remove listings (code blocks)
    body = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '', body, flags=re.DOTALL)
    body = re.sub(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', '', body, flags=re.DOTALL)
    
    # Replace common commands with their content or space
    # \section{Title} -> Title
    body = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', body)
    
    # Remove remaining backslash commands like \item, \centering, etc.
    body = re.sub(r'\\[a-zA-Z]+', ' ', body)
    
    # Remove math $$ ... $$ or $ ... $
    body = re.sub(r'\$.*?\$', ' ', body)
    
    # Remove brackets []
    body = re.sub(r'\[.*?\]', ' ', body)

    # 3. Count Words
    words = re.findall(r'\b\w+\b', body)
    
    # Filter out numbers/single chars if needed, but standard word count includes them
    valid_words = [w for w in words if len(w) > 1 or w.lower() in ['a', 'i']]
    
    print(f"Main Body Word Count: {len(valid_words)}")

if __name__ == "__main__":
    count_words_in_tex("main.tex")
