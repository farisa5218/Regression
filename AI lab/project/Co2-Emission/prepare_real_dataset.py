"""
prepare_real_dataset.py
========================
Downloads the REAL US EPA fueleconomy.gov vehicle dataset (1984-2024+) 
and converts it to our project's FuelConsumptionCo2.csv format.

Data Source: https://www.fueleconomy.gov/feg/download.shtml (official US government open data)
"""

import urllib.request
import pandas as pd
import numpy as np
import zipfile
import io
import os

def download_and_prepare():
    print("=" * 60)
    print("  Real Dataset Preparation - CO2 Emission Project")
    print("  Source: US EPA fueleconomy.gov (Official Government)")
    print("=" * 60)
    
    print("\n[1/4] Downloading official EPA dataset...")
    url = 'https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip'
    response = urllib.request.urlopen(url)
    zf = zipfile.ZipFile(io.BytesIO(response.read()))
    with zf.open(zf.namelist()[0]) as f:
        raw_df = pd.read_csv(f, encoding='latin1', low_memory=False)
    
    print(f"    Downloaded: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")
    print(f"    Year range: {raw_df['year'].min()} to {raw_df['year'].max()}")
    
    print("\n[2/4] Mapping EPA columns to project format...")
    
    # -------------------------------------------------------
    # Map EPA columns -> Our target format
    # city08    = city mpg (gasoline)
    # highway08 = highway mpg
    # co2TailpipeGpm = CO2 in grams per MILE for city
    # We want g/km, so: g/km = g/mile * 0.621371
    # Fuel consumption in L/100km from mpg: L/100km = 235.21 / mpg
    # -------------------------------------------------------
    
    # Transmission type mapping: EPA uses detailed codes
    def map_transmission(trany):
        if not isinstance(trany, str):
            return 'A6'
        t = trany.strip().lower()
        if 'automatic' in t:
            gears = ''.join(filter(str.isdigit, t))
            if gears:
                return f'AS{gears}' if len(gears) == 1 else f'A{gears}'
            return 'A6'
        elif 'manual' in t:
            gears = ''.join(filter(str.isdigit, t))
            return f'M{gears}' if gears else 'M6'
        elif 'cvt' in t or 'variable' in t:
            return 'AV'
        elif 'semi' in t:
            return 'AM6'
        return 'A6'
    
    # Fuel type mapping: EPA -> our codes
    def map_fuel_type(fueltype):
        if not isinstance(fueltype, str):
            return 'X'
        fl = fueltype.strip().lower()
        if 'premium' in fl or 'premium gasoline' in fl:
            return 'Z'
        elif 'diesel' in fl or 'biodiesel' in fl:
            return 'D'
        elif 'ethanol' in fl or 'flex' in fl or 'e85' in fl:
            return 'E'
        elif 'electric' in fl or 'hydrogen' in fl or 'natural' in fl or 'propane' in fl:
            return None  # Exclude non-combustion vehicles
        else:
            return 'X'  # Regular gasoline
    
    # Vehicle class mapping
    def map_vehicle_class(vclass):
        if not isinstance(vclass, str):
            return 'MID-SIZE'
        v = vclass.strip().lower()
        if 'compact' in v:
            return 'COMPACT'
        elif 'midsize' in v or 'mid-size' in v or 'standard' in v:
            return 'MID-SIZE'
        elif 'large' in v or 'full' in v:
            return 'FULL-SIZE'
        elif 'minicompact' in v or 'subcompact' in v:
            return 'SUBCOMPACT'
        elif 'two seater' in v or 'two-seater' in v:
            return 'TWO-SEATER'
        elif 'suv' in v and ('small' in v or 'sport' in v):
            return 'SUV - SMALL'
        elif 'suv' in v or 'sport utility' in v:
            return 'SUV - STANDARD'
        elif 'minivan' in v or 'mini van' in v:
            return 'MINIVAN'
        elif 'van' in v and 'passenger' in v:
            return 'VAN - PASSENGER'
        elif 'van' in v:
            return 'VAN - CARGO'
        elif 'pickup' in v and 'small' in v:
            return 'PICKUP TRUCK - SMALL'
        elif 'pickup' in v or 'truck' in v:
            return 'PICKUP TRUCK - STANDARD'
        elif 'station wagon' in v and 'midsize' in v:
            return 'STATION WAGON - MID-SIZE'
        elif 'station wagon' in v:
            return 'STATION WAGON - SMALL'
        elif 'special' in v:
            return 'SPECIAL PURPOSE VEHICLE'
        else:
            return 'MID-SIZE'
    
    # Filter to only vehicles with real CO2 data and gasoline/diesel engines (from 2000 onwards)
    df_work = raw_df[
        (raw_df['year'] >= 2000) &
        (raw_df['co2TailpipeGpm'] > 0) &
        (raw_df['city08'] > 0) &
        (raw_df['highway08'] > 0) &
        (raw_df['cylinders'].notna()) &
        (raw_df['displ'].notna())
    ].copy()
    
    print(f"    After filtering valid records: {df_work.shape[0]:,} rows")
    
    # Apply mappings
    df_work['FUELTYPE_MAPPED'] = df_work['fuelType1'].apply(map_fuel_type)
    df_work = df_work[df_work['FUELTYPE_MAPPED'].notna()].copy()
    
    print(f"    After removing non-combustion vehicles: {df_work.shape[0]:,} rows")
    
    # Convert units:
    # - CO2: EPA reports in g/mile for combined → convert to g/km
    #   (We use co2TailpipeGpm which is g/mile)
    # - Fuel consumption: EPA gives city/highway mpg → convert to L/100km
    #   L/100km = 235.214 / mpg
    
    df_work['CO2_GKPM'] = (df_work['co2TailpipeGpm'] / 1.60934).round(0).astype(int)
    df_work['FC_CITY']  = (235.214 / df_work['city08']).round(1)
    df_work['FC_HWY']   = (235.214 / df_work['highway08']).round(1)
    
    df_work['TRANSMISSION_MAPPED'] = df_work['trany'].apply(map_transmission)
    df_work['VEHICLECLASS_MAPPED'] = df_work['VClass'].apply(map_vehicle_class)
    
    # Build final DataFrame
    final_df = pd.DataFrame({
        'MODELYEAR':             df_work['year'].astype(int),
        'MAKE':                  df_work['make'].str.strip().str.upper(),
        'MODEL':                 df_work['model'].str.strip().str.upper(),
        'VEHICLECLASS':          df_work['VEHICLECLASS_MAPPED'],
        'ENGINESIZE':            df_work['displ'].round(1),
        'CYLINDERS':             df_work['cylinders'].astype(int),
        'TRANSMISSION':          df_work['TRANSMISSION_MAPPED'],
        'FUELTYPE':              df_work['FUELTYPE_MAPPED'],
        'FUELCONSUMPTION_CITY':  df_work['FC_CITY'],
        'FUELCONSUMPTION_HWY':   df_work['FC_HWY'],
        'FUELCONSUMPTION_COMB':  ((df_work['FC_CITY'] + df_work['FC_HWY']) / 2).round(1),
        'FUELCONSUMPTION_COMB_MPG': df_work['comb08'].astype(int),
        'CO2EMISSIONS':          df_work['CO2_GKPM']
    })
    
    # Filter out unrealistic values
    final_df = final_df[
        (final_df['CO2EMISSIONS'] >= 80) &
        (final_df['CO2EMISSIONS'] <= 600) &
        (final_df['ENGINESIZE'] >= 0.6) &
        (final_df['ENGINESIZE'] <= 10) &
        (final_df['CYLINDERS'] >= 2) &
        (final_df['CYLINDERS'] <= 16) &
        (final_df['FUELCONSUMPTION_CITY'] >= 3) &
        (final_df['FUELCONSUMPTION_CITY'] <= 50)
    ].drop_duplicates()
    
    print(f"\n[3/4] Final dataset prepared: {len(final_df):,} real vehicle records")
    print(f"    Year range: {final_df['MODELYEAR'].min()} - {final_df['MODELYEAR'].max()}")
    print(f"    Unique makes: {final_df['MAKE'].nunique()}")
    print(f"    CO2 range: {final_df['CO2EMISSIONS'].min()} - {final_df['CO2EMISSIONS'].max()} g/km")
    
    # Save the file
    output_path = 'FuelConsumptionCo2.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n[4/4] Saved to: {output_path}")
    print(f"    File size: {os.path.getsize(output_path)/1024:.1f} KB")
    print(f"\n    Sample records:")
    print(final_df[['MODELYEAR','MAKE','MODEL','ENGINESIZE','CO2EMISSIONS']].sample(8).to_string())
    print("\n✅ Real dataset preparation COMPLETE!")
    print("   Source: US EPA fueleconomy.gov (official US government open data)")
    print("   Dataset covers model years 2000-2024+")
    print("   CO2 values are EPA-measured real tailpipe emissions.")
    
    return final_df

if __name__ == '__main__':
    df = download_and_prepare()
