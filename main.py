from backend.app import app
from dotenv import load_dotenv
import os

load_dotenv()  # read .env if present, so the demo can be run from any dir
DEFAULT_PORT = 8989

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DEPLOYMENT_PORT", str(DEFAULT_PORT)))
    uvicorn.run(app, host="0.0.0.0", port=port)
