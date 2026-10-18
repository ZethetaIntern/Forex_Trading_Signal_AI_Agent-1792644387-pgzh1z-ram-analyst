
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
REPORTS_DIR = PROJECT_ROOT / "Reports"
EXCEL_DIR = PROJECT_ROOT / "Excel"

EXCEL_DIR.mkdir(parents=True, exist_ok=True)

workbook_path = EXCEL_DIR / "Forex_Trading_Signal_AI_Report.xlsx"

sheets = {
    "Price Data": DATA_DIR / "forex_price_data.csv",
    "Technical Indicators": REPORTS_DIR / "forex_technical_indicators.csv",
    "Trading Signals": REPORTS_DIR / "forex_trading_signals.csv",
    "Backtest Results": REPORTS_DIR / "forex_backtest_results.csv",
    "Model Comparison": REPORTS_DIR / "model_comparison_report.csv",
    "Walk Forward Folds": REPORTS_DIR / "xgboost_walk_forward_results.csv",
    "Walk Forward Summary": REPORTS_DIR / "xgboost_walk_forward_summary.csv",
    "XGBoost Tuning": REPORTS_DIR / "xgboost_tuning_results.csv",
    "RF Fair Comparison": REPORTS_DIR / "random_forest_fair_comparison_results.csv",
    "XGBoost Importance": REPORTS_DIR / "xgboost_feature_importance.csv",
    "RF Importance": REPORTS_DIR / "random_forest_feature_importance.csv",
    "LSTM Results": REPORTS_DIR / "lstm_results.csv",
}

with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
    for sheet_name, csv_path in sheets.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Added: {sheet_name} ({len(df)} rows)")
        else:
            print(f"Skipped missing file: {csv_path}")

    workbook = writer.book

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            max_length = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 12), 35
            )

print("\nExcel workbook created successfully:")
print(workbook_path)
