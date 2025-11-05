# main.py for create_ml_dataset Cloud Function
import functions_framework
import json
from datetime import datetime

@functions_framework.http
def create_ml_dataset(request):
    """
    Dummy Cloud Function for DAG testing.
    Simply logs invocation and returns success JSON.
    """
    ts = datetime.utcnow().isoformat()
    return (
        json.dumps({
            "status": "success",
            "message": "create_ml_dataset function executed successfully.",
            "timestamp": ts
        }),
        200,
        {"Content-Type": "application/json"},
    )
