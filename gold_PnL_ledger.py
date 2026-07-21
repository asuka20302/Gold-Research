import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
SIGNAL_FILE = PROJECT_DIR / "outputs" / "qmt_gold_signals.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "python_pnl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_CAPITAL=100000
ENTRY_THRESHOLD=0.0
TARGET_WEIGHT=0.9
FEE_RATE=0.0003
SLIPPAGE_WEIGHT=0.0002

signals=pd.read_csv(SIGNAL_FILE,encoding='gbk')
numeric_columns=[
    'predicted_return',
    'actual_return',
    'entry_open',
    'exit_close',
    'predicted_gold_price',
]

for i in numeric_columns:
    signals[i]=pd.to_numeric(signals[i], errors='coerce')
signals=signals.sort_values("execution_date").reset_index(drop=True)
capital=INITIAL_CAPITAL
ledger=[]

for index, row in signals.iterrows():
    signal_date=row['signal_date']
    entry_date=row["entry_date"]
    exit_date=row["exit_date"]
    predicted_return=row["predicted_return"]
    actual_return=row["actual_return"]
    entry_open=row["entry_open"]
    exit_close=row["exit_close"]
    predicted_gold_price=row['predicted_gold_price']
    capital_before=capital

    if predicted_return>ENTRY_THRESHOLD:
        position=1
        target_weight=TARGET_WEIGHT
    else:
        position=0
        target_weight=0.0
    if position ==1:
        entry_fill_price=entry_open*(1+SLIPPAGE_WEIGHT)
        exit_fill_price=exit_close*(1-SLIPPAGE_WEIGHT)

        capital_allocated=capital_before*target_weight
        shares=capital_allocated/entry_fill_price

        buy_value=entry_fill_price*shares
        buy_fee=buy_value*FEE_RATE

        sell_value=exit_fill_price*shares
        sell_fee=sell_value*FEE_RATE

        gross_pnl=sell_value-buy_value
        total_fee=buy_fee+sell_fee
        net_pnl=gross_pnl-total_fee
        capital_after=capital_before+net_pnl
    else:
        entry_fill_price = 0.0
        exit_fill_price = 0.0
        capital_allocated = 0.0
        shares = 0.0
        buy_value = 0.0
        sell_value = 0.0
        buy_fee = 0.0
        sell_fee = 0.0
        gross_pnl = 0.0
        total_fee = 0.0
        net_pnl = 0.0
        capital_after = capital_before

    strategy_return = capital_after / capital_before - 1
    ledger.append(
        {
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "model": row["model"],
            "horizon": row["horizon"],
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "position": position,
            "target_weight": target_weight,
            "entry_open": entry_open,
            "exit_close": exit_close,
            "entry_fill_price": entry_fill_price,
            "exit_fill_price": exit_fill_price,
            "predicted_gold_price": predicted_gold_price,
            "shares": shares,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "buy_fee": buy_fee,
            "sell_fee": sell_fee,
            "total_fee": total_fee,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "capital_before": capital_before,
            "capital_after": capital_after,
            "strategy_return": strategy_return,
        }
    )

    capital = capital_after
ledger=pd.DataFrame(ledger)
ledger["equity"] = ledger["capital_after"]
ledger["running_max_equity"] = ledger["equity"].cummax()
ledger["drawdown"] = ledger["equity"] / ledger["running_max_equity"] - 1

total_return = ledger["equity"].iloc[-1] / INITIAL_CAPITAL - 1
max_drawdown = ledger["drawdown"].min()
number_of_trades = int(ledger["position"].sum())

winning_trades = ledger[(ledger["position"] == 1) & (ledger["net_pnl"] > 0)]
losing_trades = ledger[(ledger["position"] == 1) & (ledger["net_pnl"] < 0)]

if number_of_trades > 0:
    win_rate = len(winning_trades) / number_of_trades
else:
    win_rate = 0.0

summary = pd.DataFrame(
    [
        {
            "initial_capital": INITIAL_CAPITAL,
            "final_capital": ledger["equity"].iloc[-1],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "number_of_trades": number_of_trades,
            "win_rate": win_rate,
            "total_net_pnl": ledger["net_pnl"].sum(),
            "total_fees": ledger["total_fee"].sum(),
        }
    ]
)

'''saving files below'''
ledger.to_csv(OUTPUT_DIR / "trade_ledger.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(12, 6))
plt.plot(ledger["entry_date"], ledger["equity"], label="Strategy equity")
plt.title("Gold Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Account Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
plt.close()

print("Saved trade ledger to:", OUTPUT_DIR / "trade_ledger.csv")
print("Saved summary to:", OUTPUT_DIR / "strategy_summary.csv")
print("Saved equity curve to:", OUTPUT_DIR / "equity_curve.png")

print(summary)
