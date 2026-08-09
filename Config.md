

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment - Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn Main.run:app --reload

# Run on a specific host and port
uvicorn Main.run:app --host 0.0.0.0 --port 8000 --reload

