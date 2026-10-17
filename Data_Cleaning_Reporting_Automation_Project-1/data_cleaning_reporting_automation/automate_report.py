"""
Data Cleaning & Reporting Automation
Run: python automate_report.py
Requirements: pandas, matplotlib
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_data_raw.csv"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    original_rows = len(df)
    original_columns = len(df.columns)

    # Standardize text fields
    for col in ["Name", "City"]:
        df[col] = df[col].astype("string").str.strip()
    df["City"] = df["City"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    df["Name"] = df["Name"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Convert columns to appropriate types; invalid values become missing
    for col in ["Age", "Orders", "Amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Join_Date"] = pd.to_datetime(df["Join_Date"], errors="coerce")

    # Remove duplicate records
    before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_dedup - len(df)

    # Treat impossible ages as missing; fill numeric missing values with median
    invalid_age_mask = (df["Age"] < 0) | (df["Age"] > 100)
    invalid_ages = int(invalid_age_mask.sum())
    df.loc[invalid_age_mask, "Age"] = pd.NA

    missing_before_fill = int(df.isna().sum().sum())
    for col in ["Age", "Orders", "Amount"]:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
    df["City"] = df["City"].fillna("Unknown")
    df["Name"] = df["Name"].fillna("Unknown")
    # Keep invalid dates as "Unknown" so they are explicit in the output
    df["Join_Date"] = df["Join_Date"].dt.strftime("%Y-%m-%d").fillna("Unknown")

    cleaned_path = OUT / "customer_data_cleaned.csv"
    df.to_csv(cleaned_path, index=False)

    # Create a visual summary
    plt.figure(figsize=(8, 5))
    city_counts = df["City"].value_counts()
    city_counts.plot(kind="bar")
    plt.title("Customers by City")
    plt.xlabel("City")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    chart_path = REPORTS / "customers_by_city.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    total_amount = float(df["Amount"].sum())
    avg_amount = float(df["Amount"].mean())
    total_orders = float(df["Orders"].sum())
    summary = f"""DATA CLEANING & REPORTING AUTOMATION — SUMMARY

Input file: data/customer_data_raw.csv
Output file: outputs/customer_data_cleaned.csv

DATASET OVERVIEW
- Rows before cleaning: {original_rows}
- Columns: {original_columns}
- Rows after duplicate removal: {len(df)}
- Duplicate rows removed: {duplicates_removed}
- Invalid ages corrected to missing before filling: {invalid_ages}
- Missing cells before fill/standardization: {missing_before_fill}

AUTOMATED REPORT
- Total customers after cleaning: {len(df)}
- Total orders: {total_orders:.0f}
- Total order amount: {total_amount:.2f}
- Average amount per customer: {avg_amount:.2f}
- Customers by city: {city_counts.to_dict()}

CLEANING RULES
1. Trim spaces in text columns.
2. Convert numeric fields and dates to appropriate types.
3. Remove exact duplicate rows.
4. Mark ages below 0 or above 100 as invalid, then fill missing ages with median.
5. Fill missing numeric values with each column's median.
6. Fill missing names/cities with 'Unknown'.
7. Represent invalid/missing dates as 'Unknown'.

VISUALIZATION
See reports/customers_by_city.png.
"""
    report_path = REPORTS / "automated_summary_report.txt"
    report_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Cleaned data saved to: {cleaned_path}")
    print(f"Summary report saved to: {report_path}")
    print(f"Chart saved to: {chart_path}")

if __name__ == "__main__":
    main()
