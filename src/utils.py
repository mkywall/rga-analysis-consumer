import math
import os
import json
import time
import logging
import pika
import numpy as np
import zipfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import secretmanager
from google.oauth2 import service_account

secret_store = os.environ.get("SECRET_STORE")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_cloud_secret_selfauth(secret_name):
    '''
    Get a secret from GCS secret manager using the default credentials of the environment.
    This will only work if you are running in an environment with a service account that has access to the secret.
    '''
    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=secret_name)
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Failed to access secret {secret_name} with self-authentication: {e}")
        return None


def get_credentials_from_env():
    load_dotenv('../.env')
    env_gcs_sa = os.environ.get("GCS_SA")
    if env_gcs_sa is None:
        return None

    J = json.loads(env_gcs_sa)
    with open("temp_creds.json", "w") as f:
        json.dump(J, f)

    return service_account.Credentials.from_service_account_file("temp_creds.json")


def get_secret(secret_env_var, gcs_secret_name=None, sa_creds: str = None, secret_store=secret_store):
    '''
    Running locally: set secret_env_var in the .env file.
    Running in the cloud: secrets are fetched from Google Secret Manager via the
    service account's own credentials (Workload Identity).
    '''
    load_dotenv('.env')
    secret = os.environ.get(secret_env_var)
    if secret:
        return secret
    elif gcs_secret_name is not None:
        gcs_secret_path = f"{secret_store}/{gcs_secret_name}"

        secret = get_cloud_secret_selfauth(gcs_secret_path)
        if secret is not None:
            return secret

        if sa_creds is not None:
            credentials = service_account.Credentials.from_service_account_file(sa_creds)
        else:
            credentials = get_credentials_from_env()

        if credentials is None:
            raise Exception("No credentials available to access GCS secret")

        client = secretmanager.SecretManagerServiceClient(credentials=credentials)
        response = client.access_secret_version(name=gcs_secret_path)
        return response.payload.data.decode("UTF-8")
    else:
        raise Exception(f"Secret {secret_env_var} not found in environment and no GCS secret name provided")


def setup_pika_client(host, port, pw, heartbeat = 60, blocked_connection_timeout = None,
                      max_retries = 10, retry_delay = 5):
    print("getting credentials")
    credentials = pika.PlainCredentials('admin', pw)
    parameters = pika.ConnectionParameters(host, port, '/', credentials, heartbeat=heartbeat,
                                           blocked_connection_timeout = blocked_connection_timeout)

    print("connecting")
    for attempt in range(1, max_retries + 1):
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            return(connection, channel)
        except pika.exceptions.AMQPConnectionError as e:
            if attempt == max_retries:
                logger.error(f"Could not connect to RabbitMQ after {max_retries} attempts: {e}")
                raise
            logger.warning(f"RabbitMQ connection attempt {attempt}/{max_retries} failed: {e}; "
                           f"retrying in {retry_delay}s")
            time.sleep(retry_delay)


def get_raw_data(client, raw_mfid):
    try:
        downloaded_files = client.datasets.download(raw_mfid, file_name = '.*.zip', output_dir = './')
        logger.info(f"Downloaded files: {downloaded_files}")
        extra_filt = [f for f in downloaded_files if f.endswith('.zip')]
        logger.info(f"Filtered zip files: {extra_filt}")
        data_zip = extra_filt[0]

    except Exception as e:
        logger.error(f'Error downloading zip file with client: {e}')
        return
    
    try:
        extracted_path = os.path.basename(data_zip).rstrip(".zip")
        with zipfile.ZipFile(data_zip, 'r') as zf:
            zf.extractall(extracted_path)

        subfolder_name = Path(extracted_path).stem
        if subfolder_name in os.listdir(extracted_path):
            extracted_path = f'{extracted_path}/{subfolder_name}'
        logger.info(f'Found subfolder in extract_path directory: updating directory to {extracted_path=}')

    except Exception as e:
        logger.error(f'Error unzipping file: {e}')
        return
    
    
    return data_zip, extracted_path




class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int16):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.uint8):
            return int(obj)
        if isinstance(obj, np.uint16):
            return int(obj)
        if isinstance(obj, np.uint32):
            return int(obj)
        if isinstance(obj, np.uint64):
            return int(obj)
        if isinstance(obj, datetime):
            return(str(obj.isoformat()))
        return json.JSONEncoder.default(self, obj)


def sanitize_metadata(obj):
    if isinstance(obj, dict):
        return {k: sanitize_metadata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_metadata(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return obj



