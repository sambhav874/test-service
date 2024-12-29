from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import subprocess
from pathlib import Path
from typing import Dict, Any
from rag import ContractRAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Directory setup from environment variables
INPUT_DIR = Path(os.getenv('INPUT_DIR', './contracts'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './cleaned_contracts'))
FINAL_OUTPUT_DIR = Path(os.getenv('FINAL_OUTPUT_DIR', './final_extracted_data'))

# Ensure all required directories exist
for directory in [INPUT_DIR, OUTPUT_DIR, FINAL_OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Initialize RAG system with model name from environment
model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-3.2-1B')
rag_system = ContractRAGSystem(model_name=model_name)

@app.post("/upload/")
async def upload_contract(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Receives a contract file, processes it with marker-pdf, and runs RAG analysis.
    
    Args:
        file (UploadFile): The uploaded contract file
        
    Returns:
        Dict[str, Any]: Processing results and status
        
    Raises:
        HTTPException: If processing fails at any stage
    """
    try:
        # Save uploaded file
        file_path = INPUT_DIR / file.filename
        contract_name = Path(file.filename).stem
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Process with marker-pdf
        command = [
            "marker_single",
            str(file_path),
            "--output_dir",
            str(OUTPUT_DIR),
        ]
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Create contract-specific output directory
        contract_output_dir = FINAL_OUTPUT_DIR / contract_name
        contract_output_dir.mkdir(exist_ok=True)
        
        # Run RAG analysis
        try:
            rag_system.process_contracts(str(OUTPUT_DIR))
            
            # Verify RAG output was created
            expected_output = FINAL_OUTPUT_DIR / contract_name / f"{contract_name}_final_extracted.json"
            if not expected_output.exists():
                raise Exception("RAG system did not generate expected output file")
                
            return {
                "status": "success",
                "message": f"Contract processed successfully: {file.filename}",
                "marker_output": str(OUTPUT_DIR),
                "rag_output": str(expected_output),
                "marker_logs": result.stdout.strip()
            }
            
        except Exception as rag_error:
            raise HTTPException(
                status_code=500,
                detail=f"RAG processing failed: {str(rag_error)}"
            )
            
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Marker processing failed: {e.stderr.strip() if e.stderr else 'Unknown error'}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during processing: {str(e)}"
        )
    
    finally:
        # Cleanup based on environment variable
        if os.getenv('CLEANUP_FILES', 'False').lower() == 'true':
            if file_path.exists():
                file_path.unlink()

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}