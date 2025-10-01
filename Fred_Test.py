from google.cloud import secretmanager
from fredapi import Fred

def get_secret(secret_id, project_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Example usage
project_id = "572963600069"  # replace with your GCP project ID
secret_id = "Fred_API_Key"

fred_api_key_value = get_secret(secret_id, project_id)

# Initialize FRED with your API key
fred = Fred(api_key=fred_api_key_value)

# Example: get 30-year mortgage rate
mortgage_30yr = fred.get_series("MORTGAGE30US")

print(mortgage_30yr.tail())
