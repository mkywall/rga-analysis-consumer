import math
import os
import json
import logging
import pika
import numpy as np
import zipfile
from datetime import datetime

secret_store = os.environ.get("SECRET_STORE")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_pika_client(host, port, pw, heartbeat = 60, blocked_connection_timeout = None):
    print("getting credentials")
    credentials = pika.PlainCredentials('admin', pw)
    parameters = pika.ConnectionParameters(host, port, '/', credentials, heartbeat=heartbeat,
                                           blocked_connection_timeout = blocked_connection_timeout)

    print("connecting")
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    return(connection, channel)


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
        with zipfile.ZipFile(data_zip, 'r') as zf:
            zf.extractall()

    except Exception as e:
        logger.error(f'Error unzipping file: {e}')
        return
    
    extracted_path = os.path.basename(data_zip).rstrip(".zip")
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



