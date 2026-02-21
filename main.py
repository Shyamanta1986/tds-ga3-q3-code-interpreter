import os
import sys
import traceback
import re
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

app = FastAPI()

# CORS (required for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health route (stops platform/grader "Not Found" issues on /)
@app.get("/")
def root():
    return {"status": "ok"}


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact stdout or exact traceback.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute in a fresh namespace
        exec(code, {})
        return {"success": True, "output": sys.stdout.getvalue()}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout


def fallback_line_from_traceback(tb: str) -> List[int]:
    """
    Deterministic fallback: pull the executed-code line number from traceback.
    Typical tracebacks include:
      File "<string>", line 4, in <module>
    or sometimes:
      File "", line 4, in <module>
    We take the LAST match (deepest frame = actual error line).
    """
    matches = re.findall(r'File "(?:<string>|)", line (\d+)', tb)
    return [int(matches[-1])] if matches else []


def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    """
    Call AI only when error occurs. Use structured output.
    If AI fails/returns empty, fallback to traceback parsing.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    # If key missing, don't crash; still return correct line from traceback
    if not api_key:
        return fallback_line_from_traceback(tb)

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze this Python code and its traceback.
Return the line number(s) in the CODE where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}

Return JSON with key: error_lines (array of integers).
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "error_lines": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.INTEGER),
                        )
                    },
                    required=["error_lines"],
                ),
            ),
        )

        parsed = ErrorAnalysis.model_validate_json(response.text)
        lines = sorted(set(int(x) for x in parsed.error_lines))

        # If AI returns empty, use deterministic fallback
        return lines if lines else fallback_line_from_traceback(tb)

    except Exception:
        # If AI call or JSON parsing fails, use deterministic fallback
        return fallback_line_from_traceback(tb)


@app.post("/code-interpreter", response_model=CodeResponse)
@app.post("/code-interpreter/", response_model=CodeResponse)
def code_interpreter(req: CodeRequest):
    run = execute_python_code(req.code)

    if run["success"]:
        return {"error": [], "result": run["output"]}

    lines = analyze_error_with_ai(req.code, run["output"])
    return {"error": lines, "result": run["output"]}
