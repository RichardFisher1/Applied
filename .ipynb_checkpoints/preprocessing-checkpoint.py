import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocessing_operating_data():

    df_op = pd.read_csv(r'4000 series operating data.csv')
    
    df_op.rename(columns={'LIQUID': 'LIQUID INFLOW 1', 'LIQUID.1': 'LIQUID INFLOW 2', 'LIQUID.2': 'LIQUID INFLOW 3',
                      'LIQUID.3': 'LIQUID INFLOW 4', 'LIQUID.4': 'LIQUID INFLOW 5', 'LIQUID.5': 'LIQUID INFLOW 6',
                      'GAS': 'GAS INFLOW 1', 'GAS.1': 'GAS INFLOW 2', 'GAS.2': 'GAS INFLOW 3', 'GAS.3': 'GAS INFLOW 4',
                      'OFFGAS': 'OFFGAS 1', 'OFFGAS.1': 'OFFGAS 2', 'PRESSURE': 'PRESSURE 1','PRESSURE.1': 'PRESSURE 2'},
            inplace=True)

    df_op = df_op.iloc[2:].reset_index(drop=True)

    df_op['Batch'] = df_op['Batch'].astype(int)

    df_op['Date and time'] = pd.to_datetime(df_op['Date and time'], dayfirst=True)

    liquid_cols = [col for col in df_op.columns if 'LIQUID INFLOW' in col]
    df_op[liquid_cols] = df_op[liquid_cols].apply(pd.to_numeric, errors='coerce')

    gas_cols = [col for col in df_op.columns if 'GAS INFLOW' in col]
    df_op[gas_cols] = df_op[gas_cols].apply(pd.to_numeric, errors='coerce')
    
    df_op['TOTAL LIQUID INFLOW'] = df_op[liquid_cols].sum(axis=1).round(2)
    
    return df_op

def preprocessing_product_data():

    df_p = pd.read_excel(r'4000 series product data.xlsx')

    df_p.rename(columns={'Product': 'Product (g/litre)'}, inplace=True)

    df_p = df_p.iloc[2:].reset_index(drop=True)

    df_p['Batch'] = df_p['Batch'].astype(int)
    
    df_p['Date and time'] = pd.to_datetime(df_p['Date and time'], dayfirst=True)

    
    return df_p

def add_product_to_operating(df_op, df_p):

    # for idx_op, row_op in df_op.iterrows():
        
    #     for idx_p, row_p in df_p.iterrows():

    #         if df_p['Date and time'].iloc[idx_p].date() == df_op['Date and time'].iloc[idx_op].date():
        
    #             diff = (df_op['Date and time'].iloc[idx_op] - df_p['Date and time'].iloc[idx_p])
    
    #             minutes = abs(int(diff.total_seconds() / 60))
    
    #             if minutes == 1:
    #                 df_op.at[idx_op, 'Product (g/litre)'] = df_p['Product (g/litre)'].iloc[idx_p]
    #                 break

    op_times = df_op['Date and time'].to_numpy()
    p_times  = df_p['Date and time'].to_numpy()
    p_values = df_p['Product (g/litre)'].to_numpy()

    df_op['Product (g/litre)'] = np.nan
    
    for i in range(len(op_times)):
        op_date = op_times[i].astype('datetime64[D]')
    
        for j in range(len(p_times)):
    
            if p_times[j].astype('datetime64[D]') != op_date:
                continue
    
            diff_minutes = abs((op_times[i] - p_times[j]) / np.timedelta64(1, 'm'))
    
            if diff_minutes <= 5:
                df_op.iat[i, df_op.columns.get_loc('Product (g/litre)')] = p_values[j]
                break

    return df_op

    #https://towardsdatascience.com/how-to-forecast-time-series-using-lags-5876e3f7f473/