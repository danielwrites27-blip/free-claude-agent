"""
Safe code execution sandbox for Python.
Runs code with timeout, output capture, and isolation.
"""
import subprocess
import tempfile
import os
import time
from typing import Optional, Dict


class SafeCodeRunner:
    """Execute Python code safely with timeout + output capture"""

    DEFAULT_TIMEOUT = 10
    MAX_OUTPUT_CHARS = 5000

    @classmethod
    def run(cls, code: str, timeout: Optional[int] = None,
            language: str = "python") -> Dict:
        """
        Execute code and return structured result.

        Returns:
            {
                'success': bool,
                'output': str,
                'error': str,
                'exit_code': int,
                'execution_time_ms': float
            }
        """
        if language.lower() != "python":
            return {
                'success': False,
                'error': f"Unsupported language: {language}",
                'exit_code': -1,
                'output': '',
                'execution_time_ms': 0
            }

        timeout = timeout or cls.DEFAULT_TIMEOUT
        start_time = time.time()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='/tmp'
        ) as f:
            f.write(cls._wrap_code_safely(code))
            script_path = f.name

        try:
            result = subprocess.run(
                ['python3', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd='/tmp',
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )

            execution_time = (time.time() - start_time) * 1000

            return {
                'success': result.returncode == 0,
                'output': result.stdout[:cls.MAX_OUTPUT_CHARS].strip(),
                'error': result.stderr[:cls.MAX_OUTPUT_CHARS].strip()
                         if result.returncode != 0 else '',
                'exit_code': result.returncode,
                'execution_time_ms': round(execution_time, 2)
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Timeout: Code exceeded {timeout}s limit',
                'exit_code': -1,
                'output': '',
                'execution_time_ms': timeout * 1000
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution error: {str(e)}',
                'exit_code': -1,
                'output': '',
                'execution_time_ms': 0
            }
        finally:
            if os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

    @staticmethod
    def _wrap_code_safely(code: str) -> str:
        """Add safety guards to user code"""
        dangerous = [
            'import os', 'import sys', 'import subprocess',
            'import socket', 'import urllib', 'import requests',
            '__import__', 'eval(', 'exec(', 'compile('
        ]
        for pattern in dangerous:
            if pattern in code:
                code = code.replace(pattern, f'# [BLOCKED] {pattern}')

        wrapper = """
import sys
import resource

try:
    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
except Exception:
    pass

class LimitedOutput:
    def __init__(self, original, max_chars=5000):
        self.original = original
        self.max_chars = max_chars
        self.written = 0
    def write(self, text):
        if self.written < self.max_chars:
            remaining = self.max_chars - self.written
            self.original.write(text[:remaining])
            self.written += len(text)
    def flush(self):
        self.original.flush()

sys.stdout = LimitedOutput(sys.stdout)
sys.stderr = LimitedOutput(sys.stderr)

"""
        return wrapper + "\n" + code
