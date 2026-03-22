import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocessing_operating_data():

    df_op = pd.read_csv(r'4000 series operating data.csv')

    df_op = df_op.iloc[2:].reset_index(drop=True)

    df_op['Batch'] = df_op['Batch'].astype(int)

    df_op['Date and time'] = pd.to_datetime(df_op['Date and time'], dayfirst=True)

    df_op = df_op.sort_values(['Batch', 'Date and time'])
    df_op = df_op.drop_duplicates(subset=['Batch', 'Date and time'], keep='first')
    
    liquid_cols = [col for col in df_op.columns if 'LIQUID' in col]
    df_op[liquid_cols] = df_op[liquid_cols].apply(pd.to_numeric, errors='coerce')

    gas_cols = [col for col in df_op.columns if 'GAS' in col]
    df_op[gas_cols] = df_op[gas_cols].apply(pd.to_numeric, errors='coerce')
    
    df_op['TOTAL LIQUID'] = df_op[liquid_cols].sum(axis=1).round(2)
    
    return df_op

def preprocessing_product_data():

    df_p = pd.read_excel(r'4000 series product data.xlsx')

    df_p.rename(columns={'Product': 'Product (g/litre)'}, inplace=True)

    df_p = df_p.iloc[2:].reset_index(drop=True)

    df_p['Batch'] = df_p['Batch'].astype(int)
    
    df_p['Date and time'] = pd.to_datetime(df_p['Date and time'], dayfirst=True)

    
    return df_p
    
def add_product_to_operating(df_op, df_p):
    
    # Sort by 'Date and time' FIRST to satisfy merge_asof, 
    # then 'Batch' to organize simultaneous timestamps.
    
    df_op_sorted = df_op.sort_values(['Date and time', 'Batch'])
    df_p_sorted = df_p.sort_values(['Date and time', 'Batch'])
    
    # Perform the merge_asof
    merged_df = pd.merge_asof(
        df_op_sorted, 
        df_p_sorted[['Date and time', 'Product (g/litre)', 'Batch']], 
        on='Date and time',
        by='Batch', 
        direction='nearest',
        tolerance=pd.Timedelta(minutes=1)
    )
    
    # Optional: Re-sort by Batch and Date if you want it chronologically per batch
    merged_df = merged_df.sort_values(['Batch', 'Date and time']).reset_index(drop=True)
    
    return merged_df

def duration_summary_operating_batches(df_op):
    
    batch_durations = df_op.groupby('Batch')['Date and time'].agg(['min', 'max'])
    batch_durations = batch_durations.rename(columns={'min': 'start_time', 'max': 'end_time'})
    batch_durations['duration'] = batch_durations['end_time'] - batch_durations['start_time']
    
    # Extract the total days as an integer and return the Series
    return batch_durations['duration'].dt.days