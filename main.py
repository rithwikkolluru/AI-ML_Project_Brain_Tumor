"""
Brain Tumor AI Diagnostic Suite — Application Entrypoint
Suitable for local execution and Render cloud deployment.
"""

import os
import uvicorn
from backend.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[INFO] Launching Brain Tumor AI Clinical Workstation on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False)
