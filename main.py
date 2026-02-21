import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def execute_python_code(code: str):
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

def analyze_error_with_ai(code: str, tb: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze this Python code and its traceback.
Return the line numbers in the CODE where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}
"""

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

    try:
        parsed = ErrorAnalysis.model_validate_json(response.text)
        return sorted(set(parsed.error_lines))
    except Exception:
        return []

@app.post("/code-interpreter")
@app.post("/code-interpreter/")
def code_interpreter(req: CodeRequest):
    run = execute_python_code(req.code)

    if run["success"]:
        return {"error": [], "result": run["output"]}

    lines = analyze_error_with_ai(req.code, run["output"])
    return {"error": lines, "result": run["output"]}
