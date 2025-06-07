# Load environment variables
from dotenv import load_dotenv
env_res = load_dotenv()

if not env_res :
    raise Exception("No .env file provided")

import os

# OAuth
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

def getOAuth() :
    # Client credentials
    client_id = os.environ["CLIENT-ID"]
    client_secret = os.environ["CLIENT-SECRET"]

    # Create session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Get token
    token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                            client_secret=client_secret,
                            include_client_id=True)

    # Proper error handling
    def sentinelhub_compliance_hook(response):
        response.raise_for_status()
        return response

    oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

    SessionData = (oauth, token)
    return SessionData

# Test
# if __name__ == "__main__" :
#     res = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
#     print(f"Response: {res.content}")