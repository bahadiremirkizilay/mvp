import pandas as pd

excel_path = 'data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'

print("=" * 80)
print("SAMM Excel File Structure Analysis")
print("=" * 80)

# Try different skiprows
for skip in [0, 8, 10, 12, 15, 18, 20]:
    print(f"\n=== skiprows={skip} ===")
    try:
        df = pd.read_excel(excel_path, skiprows=skip, nrows=5)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:8]}")
        print(f"First row data: {df.iloc[0].tolist()[:5]}")
        
        # Check if this looks like the actual data
        if 'Subject' in str(df.columns) or 'subject' in str(df.columns).lower():
            print("✅ FOUND HEADERS!")
            break
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 80)
