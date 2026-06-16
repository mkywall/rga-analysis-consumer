import os
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from crucible import BaseDataset
from crucible_utils.general_utils import run_shell, get_utc_isoformat
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

num_cores = os.cpu_count()

# get raw data from crucible
def get_raw_data(client, raw_mfid):
    try:
        downloaded_files = client.datasets.download(raw_mfid, file_name = '.*.zip', output_dir = './')
        logger.info(f"Downloaded files: {downloaded_files}")
        extra_filt = [f for f in downloaded_files if f.endswith('.zip')]
        logger.info(f"Filtered zip files: {extra_filt}")
        zipfile = extra_filt[0]
    except Exception as e:
        logger.error(f'Error downloading zip file with client: {e}')
        return
    
    try:
        extracted_path = os.path.basename(zipfile).rstrip(".zip")
        if not os.path.exists(extracted_path):
            run_shell(f"unzip '{zipfile}'")
    except Exception as e:
        logger.error(f'Error unzipping file: {e}')
        return
    
    extracted_path = os.path.basename(zipfile).rstrip(".zip")
    return zipfile, extracted_path


def create_dataset_metadata(og_dataset, sample_results):
    _, __, sample = sample_results

    # Create the Dataset Object 
    og_name = og_dataset['dataset_name']
    raw_measurement = og_dataset['measurement']
    new_dataset = {'dataset_name':f"{og_name} - {sample} - aggregated", 
                   'measurement': f'aggregated {raw_measurement} data', 
                   'timestamp':get_utc_isoformat(),
                   'data_type': f'aggregated {raw_measurement} csv files',
                   'data_format':'csv'}
    
    # copy over relevant fields from parent
    og_fields = ["owner_orcid", 'project_id', "instrument_name", "public", "session_name"]
    
    for field in og_fields:
        if field in og_dataset:
            new_dataset[field] = og_dataset[field]
        else:
            logger.warning(f"Missing field '{field}' in original dataset")

    crux_dataset = BaseDataset(**new_dataset)

    # scientific metadata
    scimd = {"github_repository": os.environ.get('REPO'),
             "git_commit_hash": os.environ.get('GITHASH'),
             "sample": sample,
             "raw_parent_dataset_name": og_name,
             'raw_parent_dataset_id': og_dataset['unique_id']
             }

    keywords = [f'aggregated {raw_measurement}']
    return crux_dataset, scimd, keywords


def upload_and_link_dataset(client, sample_results, ds, scimd, kw, og_dataset):
    files_to_upload, plot, sample = sample_results
    try:
        assert ds.project_id is not None
    except Exception as e:
        logger.error(f'Dataset Project_id is None - skipping: {e}')
        return
    
    # create the dataset
    new_ds = client.datasets.create(ds,
                                    files_to_upload = files_to_upload,
                                    scientific_metadata = scimd,
                                    keywords = kw
                                    )
    new_dsid = new_ds['created_record']['unique_id']


    # add thumbnails
    client.datasets.add_thumbnail(new_dsid, image = plot,
                                  thumbnail_name = f'{sample} 2D Plot')
    
    # link to parent dataset
    client.datasets.link_parent_child(parent_dataset_id = og_dataset.get('unique_id'),
                                      child_dataset_id = new_dsid)
        
    # link to samples
    found_samps = client.samples.list(sample_name = sample)
    if len(found_samps) == 1:
        samp_id = found_samps[-1]['unique_id']
        client.samples.add_to_dataset(dataset_id = new_dsid, sample_id = samp_id)
    else:
        logger.warning(f'Found samples with name {sample}: {found_samps}')
    
    return new_ds
















def rga_analysis(client, raw_mfid):
        # get the dataset
        try:
            og_dataset = client.datasets.get(raw_mfid)
            raw_measurement = og_dataset['measurement']
        except Exception as e:
            logger.error(f"Error retrieving dataset {raw_mfid}: {e}")
            return
        
        # get the raw data 
        try:
            zipfile, extracted_path = get_raw_data(client, raw_mfid)
        except Exception as e:
            logger.error(f"Error copying and extracting zipfile for {raw_mfid}: {e}")
            return

        # Process samples in parallel using multiprocessing
        try:
            sample_folders = identify_insitu_samples(extracted_path)
            num_jobs = int(0.5*num_cores)
            results = Parallel(n_jobs=num_jobs)(delayed(process_sample)(i,raw_measurement) for i in sample_folders.items())
        except Exception as e:
            logger.error(f'Error processing samples: {e}')
            return
        

        # Create the Dataset in Crucible
        for sample_results in results:
            try:
                crux_dataset, scimd, keywords = create_dataset_metadata(og_dataset, sample_results)
            except Exception as e:
                logger.error(f'Error creating dataset metadata: {e}')
                return 
            
            try:
                upload_and_link_dataset(client, sample_results, crux_dataset, scimd, keywords, og_dataset )
            except Exception as e:
                logger.error(f'Error uploading results: {e}')
                return

