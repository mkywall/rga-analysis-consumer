
print("importing packages")
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import json

from crucible import CrucibleClient
from crucible.models import Dataset
import mfid
from datetime import datetime, timezone

import threading
import rga_batch
from utils import setup_pika_client, get_raw_data, get_secret, sanitize_metadata
from dotenv import load_dotenv
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Vars ===========================
load_dotenv()
rmq_host = os.environ.get("RMQ_HOST", "localhost")
rmq_port = int(os.environ.get("RMQ_PORT", 5672))
rmq_pw = get_secret("RABBITMQ_DEFAULT_PW", "rabbitmq_default_pw/versions/1")

crucible_api_url = os.environ.get("CRUCIBLE_API_URL", "https://crucible.lbl.gov/api/v2")
crucible_api_key = get_secret("ADMIN_APIKEY", "crucible_admin_apikey/versions/4")

num_cores = os.cpu_count()
print(f"{num_cores=}")

# RMQ Setup ===========================
connection, channel = setup_pika_client(rmq_host, rmq_port, rmq_pw)
queues_needed = ['rga-analysis', 'rga-analysis-failed']

for q in queues_needed:
    channel.queue_declare(queue=q)

# Crucible  ===========================
client = CrucibleClient(api_url=crucible_api_url, api_key=crucible_api_key)


# Functions  ===========================
def fetch_existing_child_map(crucible_client, parent_id):
    def fetch_sample(sds):
        sample = crucible_client.samples.list(dataset_id=sds["unique_id"])[0]
        return sample["sample_name"], sds["unique_id"]

    children = crucible_client.datasets.list_children(parent_id)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_sample, children)
    return dict(results)


def is_empty_spot(sample_name):
    return sample_name is None or str(sample_name).strip().lower() in ("", "nan", "none")


def validate_sample_files(sample_dictionary, directory):
    """
    Check every sample on the holder has exactly one TEY and one RGA file before
    anything is uploaded, so a bad batch fails with no partial children created.

    Returns the set of sample names that were measured in this batch.
    """
    measured = set()
    invalid = []

    for spot, entry in sample_dictionary.items():
        sample_name = entry["sample_name"]
        if is_empty_spot(sample_name):
            continue

        tey_files, rga_files = rga_batch.match_sample_files(sample_name, directory)

        if not tey_files and not rga_files:
            logger.info(f"[{spot} {sample_name}] not measured in this batch, skipping")
        elif len(tey_files) == 1 and len(rga_files) == 1:
            measured.add(sample_name)
        else:
            invalid.append((spot, sample_name, tey_files, rga_files))

    if invalid:
        lines = ["Expected exactly 1 TEY and 1 RGA file per sample, found:"]
        for spot, sample_name, tey_files, rga_files in invalid:
            lines.append(f"  [{spot}] {sample_name}: {len(tey_files)} TEY, {len(rga_files)} RGA")
            for f in tey_files + rga_files:
                lines.append(f"    {os.path.basename(f)}")
        raise ValueError("\n".join(lines))

    return measured


def create_sample_dataset(sample_entry, spot, ds, directory, crucible_client, sample_sub_dataset_id_map, measured_samples):
    sample_id = sample_entry["sample_id"]
    sample_name = sample_entry["sample_name"]

    # Skip empty/placeholder spots so we don't create datasets with no sample name.
    if is_empty_spot(sample_name):
        logger.warning(f"[{spot}] No sample name for this spot, skipping dataset creation")
        return None

    # Skip spots with no data files so we don't create empty datasets.
    if sample_name not in measured_samples:
        logger.warning(f"[{spot} {sample_name}] No files found for sample, skipping dataset creation")
        return None

    m = rga_batch.process_sample(sample_name, directory=directory, overwrite=True)
    sample_files = m.files

    sds_mfid = sample_sub_dataset_id_map.get(sample_name, mfid.mfid()[0])

    sds = Dataset(unique_id = sds_mfid,
                  dataset_name = f"RGATEY_{ds['dataset_name']}_{spot}_{sample_name}",
                  instrument_name = "ALS-BL12012",
                  measurement = "automated_RGA_TEY_run", # TODO - swap to RGA/TEY?
                  project_id = ds['project_id'],   # use project_id of the parent
                  data_type = "automated_RGA_TEY_run")

    # Timestamp off the raw RGA file: the derived files were written moments ago and
    # would record the processing time rather than the measurement time.
    rga_raw = os.path.join(directory, m.metadata["rga_source"])
    sds.timestamp = datetime.fromtimestamp(os.path.getmtime(rga_raw), tz=timezone.utc).isoformat()

    crucible_client.datasets.create(sds, files_to_upload=sample_files, scientific_metadata = sanitize_metadata(m.metadata), wait_for_ingestion_response=False)

    crucible_client.datasets.link_parent_child(ds['unique_id'], sds.unique_id)
    crucible_client.samples.add_dataset(sample_id, sds.unique_id)

    # Record the corrected outgassing area on the sample's scientific metadata
    outgas_area = m.outgas_area
    if outgas_area is not None:
        md = {
            'outgas_area': float(outgas_area),
            'outgas_area_analysis_reference': f'dataset: {sds.unique_id}',
        }
        crucible_client.samples.update_scientific_metadata(sample_id, metadata=md)
        logger.info(f"  [{spot}] {sample_name} outgas_area={outgas_area:.3e} → sample {sample_id}")
    else:
        logger.warning(f"  [{spot}] {sample_name} no outgas_area found, skipping metadata update")

    if m.thumbnail:
        tn_name = os.path.basename(m.thumbnail)
        try:
            with open(m.thumbnail, "rb"):
                pass
            crucible_client.datasets.add_thumbnail(dsid=sds_mfid, image=m.thumbnail, thumbnail_name=tn_name)
        except FileNotFoundError:
            logger.warning(f"  [warn] thumbnail not found, skipping: {m.thumbnail}")

    logger.info(f"  [{spot}] {sample_name} → {sds.unique_id}")
    return sample_name, sds_mfid


def run_rga_analysis(ch, method, body, connection):
    message = json.loads(body.decode("utf-8").strip())
    raw_mfid  = message['dsid']

    print(f"received message {message} .. starting processing")

    data_zip = None
    directory = None

    try:
        # get the dataset SQL record
        og_dataset = client.datasets.get(raw_mfid, include_metadata=True)

        # get the raw data files
        data_zip, directory = get_raw_data(client, raw_mfid)

        sample_dictionary = og_dataset['scientific_metadata']['samples']
        sample_positions = list(sample_dictionary.keys())

        # Validate before anything is uploaded, so a bad batch creates no children.
        measured_samples = validate_sample_files(sample_dictionary, directory)
        logger.info(f"{len(measured_samples)} samples measured in this batch")

        # upload to crucible -  following Ed's workflow
        logger.info('Fetching existing child map...')
        sample_sub_dataset_id_map = fetch_existing_child_map(client, raw_mfid)

        logger.info(f"Analyzing and creating sample datasets...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            child_results = list(executor.map(
                        lambda sample_position: create_sample_dataset(sample_dictionary[sample_position],
                                                                      sample_position, 
                                                                      og_dataset,
                                                                      directory,
                                                                      client,
                                                                      sample_sub_dataset_id_map,
                                                                      measured_samples),
                        sample_positions,
                    ))

        logger.info(f'Crucible upload complete.')
        # acknowledge the message                                                                          
        connection.add_callback_threadsafe(lambda: ch.basic_ack(delivery_tag=method.delivery_tag))     
    
    except Exception as err:
        logger.error(f"Error processing dataset {raw_mfid}: {err}")
        def on_failure():
            ch.basic_publish(                                                                                                                          
                exchange='',                                                                                                                           
                routing_key='rga-analysis-failed',                                                                                               
                body=json.dumps(message)                                                                                                               
            )                                                                                                                                          
            ch.basic_ack(delivery_tag=method.delivery_tag)

        connection.add_callback_threadsafe(on_failure)

    finally:
        # Clean up the downloaded zip and extracted folder regardless of outcome
        if data_zip and os.path.isfile(data_zip):
            try:
                os.remove(data_zip)
                logger.info(f"Removed zip file: {data_zip}")
            except OSError as e:
                logger.warning(f"Could not remove zip file {data_zip}: {e}")

        if data_zip:
            # get_raw_data may return a nested subfolder, so delete the top-level
            # extraction root derived from the zip name.
            extract_root = os.path.basename(data_zip).removesuffix(".zip")
            if os.path.isdir(extract_root):
                try:
                    shutil.rmtree(extract_root)
                    logger.info(f"Removed extracted folder: {extract_root}")
                except OSError as e:
                    logger.warning(f"Could not remove extracted folder {extract_root}: {e}")


def callback(ch, method, properties, body):
    '''
    Expects a RMQ message with: 
    dsid:     The dataset ID that the processing request was made for
              and that the new data will be uploaded to

    '''
    thread = threading.Thread(target=run_rga_analysis, args=(ch, method, body, connection))                                                                                   
    thread.start()                                                                                                                                 
    

# subscribe to the queue
channel.basic_consume(queue='rga-analysis',
                      auto_ack=False,
                      on_message_callback=callback)

# always be listening
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()


























