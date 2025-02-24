import json
import pandas as pd
from datasets import load_dataset
import psutil
import time
from IPython.display import clear_output
import os


def get_first_review_dates(batch_size=1000, category="Books"):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                          f"raw_review_{category}", 
                          streaming=True,
                          trust_remote_code=True)
    
    start_time = time.time()
    records_processed = 0
    first_review_dates = {}
    
    # Process in batches
    for batch in dataset["full"].iter(batch_size=batch_size):
        # Convert batch to DataFrame
        df_batch = pd.DataFrame(batch)
        
        # Convert timestamp
        df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')
        
        # Update first review dates
        for asin, group in df_batch.groupby('asin'):
            min_date = group['timestamp'].min()
            if asin not in first_review_dates:
                first_review_dates[asin] = min_date
            else:
                first_review_dates[asin] = min(first_review_dates[asin], min_date)
        
        # Update counts
        records_processed += len(df_batch)
        elapsed_time = time.time() - start_time
        
        # Update display every 10,000 records
        if records_processed % 10000 == 0:
            clear_output(wait=True)
            print(f"{'='*40}")
            print(f"Records Processed: {records_processed:,}")
            print(f"Unique Products: {len(first_review_dates):,}")
            print(f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"Time Elapsed: {elapsed_time:.1f} seconds")
            print(f"Processing Rate: {records_processed/elapsed_time:.1f} records/second")
            print(f"{'='*40}")
            
            # Save periodically
            if records_processed % 100000 == 0:
                os.makedirs(f'../../data/{category}', exist_ok=True)
                temp_df = pd.DataFrame.from_dict(first_review_dates, 
                                              orient='index', 
                                              columns=['first_review_date'])
                temp_df.index.name = 'asin'
                temp_df.to_parquet(
                    f'../../data/{category}/first_review_dates.parquet'
                )
    
    # Save final results
    os.makedirs(f'../../data/{category}', exist_ok=True)
    final_df = pd.DataFrame.from_dict(first_review_dates, 
                                    orient='index', 
                                    columns=['first_review_date'])
    final_df.index.name = 'asin'
    final_df.to_parquet(f'../../data/{category}/first_review_dates_final.parquet')
    
    return first_review_dates

def process_filtered_dataset(batch_size=10000, category="Books", asins_to_keep=None):
    # Convert asins_to_keep to a set for faster lookup
    asins_set = set(asins_to_keep)
    
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                          f"raw_review_{category}", 
                          streaming=True,
                          trust_remote_code=True)
    
    start_time = time.time()
    records_processed = 0
    processed_chunks = []
    
    # Process in larger batches
    for batch in dataset["full"].iter(batch_size=batch_size):
        # Convert batch to DataFrame more efficiently
        df_batch = pd.DataFrame(batch)
        
        # Filter using isin with set
        mask = df_batch['asin'].isin(asins_set)
        if mask.any():
            df_filtered = df_batch[mask].copy()
            # Convert timestamp in one go
            df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], unit='ms')
            processed_chunks.append(df_filtered)
        
        records_processed += len(df_batch)
        
        # Save more frequently with larger chunks
        if records_processed % 100000 == 0:
            clear_output(wait=True)
            print(f"{'='*40}")
            print(f"Records Processed: {records_processed:,}")
            print(f"Filtered Records: {sum(len(chunk) for chunk in processed_chunks):,}")
            print(f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"Processing Rate: {records_processed/(time.time() - start_time):.1f} records/second")
            print(f"Time Elapsed: {time.time() - start_time:.1f} seconds")
            print(f"{'='*40}")
            
            if processed_chunks:
                os.makedirs(f'../../data/{category}_23', exist_ok=True)
                pd.concat(processed_chunks).to_parquet(
                    f'../../data/{category}_23/filtered_data_{records_processed}.parquet'
                )
                processed_chunks = []
    
    # Save remaining chunks
    if processed_chunks:
        os.makedirs(f'../../data/{category}_23', exist_ok=True)
        pd.concat(processed_chunks).to_parquet(
            f'../../data/{category}_23/filtered_data_remaining.parquet'
        )
        print(f"Final chunks saved with {sum(len(chunk) for chunk in processed_chunks)} records")



def load_local_jsonl(file_path, batch_size=1000):
    """Generator function to load JSONL file in batches"""
    batch = []
    with open(file_path, 'r') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield pd.DataFrame(batch)
                batch = []
    if batch:  # Yield the last partial batch
        yield pd.DataFrame(batch)

def get_first_review_dates_local(file_path, batch_size=1000, category="electronics", timestamp_col="timestamp", version = 2023):
    """Modified version of get_first_review_dates for local JSONL file"""
    start_time = time.time()
    records_processed = 0
    first_review_dates = {}
    
    # Process in batches
    for df_batch in load_local_jsonl(file_path, batch_size):
        # Convert timestamp (adjust the column name if different in your JSON)
        if version == 2023:
            df_batch[timestamp_col] = pd.to_datetime(df_batch[timestamp_col], unit='ms')
        else:
            df_batch[timestamp_col] = pd.to_datetime(df_batch[timestamp_col])
        
        # Update first review dates
        for asin, group in df_batch.groupby('asin'):
            min_date = group[timestamp_col].min()
            if asin not in first_review_dates:
                first_review_dates[asin] = min_date
            else:
                first_review_dates[asin] = min(first_review_dates[asin], min_date)
        
        # Update counts
        records_processed += len(df_batch)
        elapsed_time = time.time() - start_time
        
        # Update display every 10,000 records
        if records_processed % 10000 == 0:
            clear_output(wait=True)
            print(f"{'='*40}")
            print(f"Records Processed: {records_processed:,}")
            print(f"Unique Products: {len(first_review_dates):,}")
            print(f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"Time Elapsed: {elapsed_time:.1f} seconds")
            print(f"Processing Rate: {records_processed/elapsed_time:.1f} records/second")
            print(f"{'='*40}")
            
            # Save periodically
            if records_processed % 100000 == 0:
                os.makedirs(f'../../data/{category}', exist_ok=True)
                temp_df = pd.DataFrame.from_dict(first_review_dates, 
                                              orient='index', 
                                              columns=['first_review_date'])
                temp_df.index.name = 'asin'
                temp_df.to_parquet(
                    f'../../data/{category}/first_review_dates.parquet'
                )
    
    # Save final results
    os.makedirs(f'../../data/{category}', exist_ok=True)
    final_df = pd.DataFrame.from_dict(first_review_dates, 
                                    orient='index', 
                                    columns=['first_review_date'])
    final_df.index.name = 'asin'
    final_df.to_parquet(f'../../data/{category}/first_review_dates_final.parquet')
    
    return first_review_dates

def process_filtered_dataset_local(file_path, batch_size=10000, asins_to_keep=None, category="electronics", timestamp_col="timestamp", version = 2023):
    """Modified version of process_filtered_dataset for local JSONL file"""
    asins_set = set(asins_to_keep)
    
    start_time = time.time()
    records_processed = 0
    processed_chunks = []
    
    # Process in batches
    for df_batch in load_local_jsonl(file_path, batch_size):
        # Filter using isin with set
        mask = df_batch['asin'].isin(asins_set)
        if mask.any():
            df_filtered = df_batch[mask].copy()
            if version == 2023:
                df_filtered[timestamp_col] = pd.to_datetime(df_filtered[timestamp_col], unit='ms')
            else:
                df_filtered[timestamp_col] = pd.to_datetime(df_filtered[timestamp_col])
            processed_chunks.append(df_filtered)
        
        records_processed += len(df_batch)
        
        if records_processed % 100000 == 0:
            clear_output(wait=True)
            print(f"{'='*40}")
            print(f"Records Processed: {records_processed:,}")
            print(f"Filtered Records: {sum(len(chunk) for chunk in processed_chunks):,}")
            print(f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"Processing Rate: {records_processed/(time.time() - start_time):.1f} records/second")
            print(f"Time Elapsed: {time.time() - start_time:.1f} seconds")
            print(f"{'='*40}")
            
            if processed_chunks:
                os.makedirs(f'../../data/{category}', exist_ok=True)
                pd.concat(processed_chunks).to_parquet(
                    f'../../data/{category}/filtered_data_{records_processed}.parquet'
                )
                processed_chunks = []
    
    # Save remaining chunks
    if processed_chunks:
        os.makedirs(f'../../data/{category}', exist_ok=True)
        pd.concat(processed_chunks).to_parquet(
            f'../../data/{category}/filtered_data_remaining.parquet'
        )
        print(f"Final chunks saved with {sum(len(chunk) for chunk in processed_chunks)} records")