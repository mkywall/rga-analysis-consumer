
print("importing packages")
from concurrent.futures import ThreadPoolExecutor
import os
import json

from crucible import CrucibleClient
from crucible.models import Dataset
import mfid
import glob
from datetime import datetime, timezone

import threading
from .utils import setup_pika_client, get_raw_data
from dotenv import load_dotenv
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Vars ===========================
load_dotenv()
rmq_host = os.environ.get("RABBITMQ_HOST", "localhost")
rmq_port = int(os.environ.get("RABBITMQ_PORT", 5672))
rmq_pw = os.environ.get("RABBITMQ_DEFAULT_PW", "rabbitmq_default_pw/versions/1")

crucible_api_url = os.environ.get("CRUCIBLE_API_URL", "https://crucible.lbl.gov/api/v2")
crucible_api_key = os.environ.get("CRUCIBLE_ADMIN_APIKEY", "crucible_admin_apikey/versions/4")

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


def create_sample_dataset(sample_entry, spot, ds, directory, crucible_client, sample_sub_dataset_id_map):
    sample_id = sample_entry["sample_id"]
    sample_name = sample_entry["sample_name"]

    sds_mfid = sample_sub_dataset_id_map.get(sample_name, mfid.mfid()[0])

    sds = Dataset(unique_id = sds_mfid,
                  dataset_name = f"RGATEY_{ds.dataset_name}_{spot}_{sample_name}",
                  instrument_name = "ALS-BL12012",
                  measurement = "automated_RGA_TEY_run", # TODO - swap to RGA/TEY?
                  project_id = "10k_perovskites",
                  data_type = "automated_RGA_TEY_run")

    sample_files = glob.glob(os.path.join(directory, f"{sample_name}_*.txt"))
    for results_dir in ("Analysis_results-ascii", "Analysis_results-plots"):
        sample_files += glob.glob(
            os.path.join(directory, results_dir, "**", f"{sample_name}_*"),
            recursive=True,
        )

    # Set the timestamp for the sample dataset based on the modification time of the first file
    if sample_files:
        sds.timestamp = datetime.fromtimestamp(os.path.getmtime(sample_files[0]), tz=timezone.utc).isoformat()
    else:
        logger.warning(f"[{spot} {sample_name}] No files found for sample dataset, timestamp will not be set")


    crucible_client.datasets.create(sds, files_to_upload=sample_files, wait_for_ingestion_response=False)

    crucible_client.datasets.link_parent_child(ds.unique_id, sds.unique_id)
    crucible_client.samples.add_dataset(sample_id, sds.unique_id)

    thumbnail_names = [
        f"MS/{sample_name}_MS_log.png",
        f"MS(t)_averaged/{sample_name}_MS_t_averaged.png",
        f"TEY_normalized/{sample_name}_TEY_normalized.png",
        f"TEY_normalized_averaged/{sample_name}_TEY_normalized_averaged.png",
        f"Total_outgassing_averaged/{sample_name}_total_outgassing_averaged.png",
    ]
    for tn_name in thumbnail_names:
        tn_path = os.path.join(directory, "Analysis_results-plots", tn_name)
        try:
            with open(tn_path, "rb"):
                pass
            crucible_client.datasets.add_thumbnail(dsid=sds_mfid, image=tn_path, thumbnail_name=tn_name)
        except FileNotFoundError:
            logger.warning(f"  [warn] thumbnail not found, skipping: {tn_path}")

    logger.info(f"  [{spot}] {sample_name} → {sds.unique_id}")
    return sample_name, sds_mfid


def run_rga_analysis(ch, method, body, connection):
    message = json.loads(body.decode("utf-8").strip())
    raw_mfid  = message['dsid']

    print(f"received message {message} .. starting processing")
    
    try:
        # get the dataset SQL record
        og_dataset = client.datasets.get(raw_mfid, include_metadata=True)

        # get the raw data files
        data_zip, directory = get_raw_data(client, raw_mfid)

        # run Kas's analysis script
        import automated_RGA_TEY_kas_20260126
        automated_RGA_TEY_kas_20260126.main(directory)

        # upload to crucible -  following Ed's workflow
        sample_sub_dataset_id_map = fetch_existing_child_map(client, raw_mfid)
        sample_dictionary = og_dataset['scientific_metadata']['samples']
        sample_positions = list(sample_dictionary.keys())

        with ThreadPoolExecutor(max_workers=8) as executor:
            child_results = list(executor.map(
                        lambda sample_position: create_sample_dataset(sample_dictionary[sample_position], sample_position, og_dataset, directory, client, sample_sub_dataset_id_map),
                        sample_positions,
                    ))

        # acknowledge the message                                                                          
        connection.add_callback_threadsafe(lambda: ch.basic_ack(delivery_tag=method.delivery_tag))     
    
    except Exception as err:
        def on_failure():
            ch.basic_publish(                                                                                                                          
                exchange='',                                                                                                                           
                routing_key='rga-analysis-failed',                                                                                               
                body=json.dumps(message)                                                                                                               
            )                                                                                                                                          
            ch.basic_ack(delivery_tag=method.delivery_tag)                                                  
        
        connection.add_callback_threadsafe(on_failure)   


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


























