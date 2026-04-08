import sys
import os
# Add the root directory to path so it can find models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from models import EmailTriageAction, EmailTriageObservation
from server.email_triage_env_environment import EmailTriageEnvironment

app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage_env",
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
