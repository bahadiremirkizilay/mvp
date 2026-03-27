import pandas as pd

excel_path = 'data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'

# Read with skiprows=12
df = pd.read_excel(excel_path, skiprows=12)

print("Original columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
