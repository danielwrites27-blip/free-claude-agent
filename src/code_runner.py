import subprocess, tempfile, os, time
from typing import Dict, Optional

class SafeCodeRunner:
    @classmethod
    def run(cls, code: str, timeout: int = 10, language: str = "python") -> Dict:
        if language != "python":
            return {"success": False, "error": f"Unsupported: {language}", "output": ""}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            f.write(code); path = f.name
        try:
            start = time.time()
            result = subprocess.run(['python3', path], capture_output=True, text=True, timeout=timeout, cwd='/tmp')
            return {"success": result.returncode==0, "output": result.stdout[:2000], "error": result.stderr[:500], "execution_time_ms": (time.time()-start)*1000}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timeout >{timeout}s", "output": ""}
        finally:
            if os.path.exists(path): os.unlink(path)
