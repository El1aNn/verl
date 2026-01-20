import pandas as pd
import numpy as np

pi1_path = '/root/rl/verl/data/dsr_sub/pi1_r128.parquet'
train_path = '/root/rl/verl/data/dsr_sub/train.parquet'

try:
    pi1 = pd.read_parquet(pi1_path)
    train = pd.read_parquet(train_path)
    
    target_size = len(train)
    current_size = len(pi1)
    
    print(f"Current size: {current_size}, Target size: {target_size}")
    
    if current_size != target_size:
        # Calculate how many times to repeat
        repeats = (target_size // current_size) + 1
        padded_pi1 = pd.concat([pi1] * repeats, ignore_index=True)
        # Truncate to exact size
        padded_pi1 = padded_pi1.iloc[:target_size]
        
        print(f"New size: {len(padded_pi1)}")
        
        # Save back to the same file
        padded_pi1.to_parquet(pi1_path)
        print(f"Successfully updated {pi1_path}")
    else:
        print("Sizes are already equal. No action taken.")

except Exception as e:
    print(f"An error occurred: {e}")
    # Fallback to pyarrow if pandas fails for some reason during write
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        pi1_table = pq.read_table(pi1_path)
        train_table = pq.read_table(train_path)
        
        target_size = train_table.num_rows
        current_size = pi1_table.num_rows
        
        if current_size != target_size:
            repeats = (target_size // current_size) + 1
            # Concatenate tables
            tables = [pi1_table] * repeats
            padded_table = pa.concat_tables(tables)
            # Slice to exact size
            padded_table = padded_table.slice(0, target_size)
            
            pq.write_table(padded_table, pi1_path)
            print(f"Successfully updated {pi1_path} using pyarrow")
    except Exception as e2:
        print(f"Pyarrow fallback also failed: {e2}")
