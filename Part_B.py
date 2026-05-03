import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#2f3b45",
    "axes.labelcolor": "#1f2933",
    "xtick.color": "#1f2933",
    "ytick.color": "#1f2933",
    "grid.color": "#cfd8dc",
    "grid.alpha": 0.35,
    "font.size": 10,
})

df = pd.read_csv(
    "Data.csv",
    header=[0, 1],
    index_col=0
)

df        = df.apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
df_clean  = df.ffill().bfill()
prices    = df_clean.xs("Close", axis=1, level=1)

# RSI tells if the stock has moved too much up or down recently.
# Low RSI is used as the buy side of this mean reversion idea.
def compute_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:

    delta    =  prices.diff()
    gains    =  delta.clip(lower=0)
    losses   = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period,  min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    rsi[(avg_loss == 0) & (avg_gain > 0)]  = 100.0
    return rsi

# Volatility shows how much the price is jumping around.
# Here it is the rolling standard deviation of price changes.
def compute_volatility(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    return prices.diff().rolling(window=period, min_periods=period).std()


# This avoids trades when the market is too jumpy.
# A trade is allowed only when volatility is below its own average.
def compute_volatility_filter(volatility: pd.DataFrame, window: int) -> pd.DataFrame:
    vol_ma = volatility.rolling(window=window, min_periods=window).mean()
    return volatility < vol_ma

rsi        = compute_rsi(prices, period=14)
volatility = compute_volatility(prices, period=14)

# Entry needs low RSI and calm volatility.
# Exit happens when RSI moves back above the upper level.
def generate_signals(prices, rsi, volatility, L, H, W):
    vol_filter = compute_volatility_filter(volatility, window=W)
    entries    = ((rsi < L) & vol_filter).shift(1).fillna(False).astype(bool)
    exits      = (rsi > H).shift(1).fillna(False).astype(bool)
    return entries, exits

# Price, entries, and exits must line up row by row.
# This removes unusable rows before sending data to vectorbt.
def align_data(*dataframes):
    common_idx = dataframes[0].index
    for df in dataframes[1:]:
        common_idx = common_idx.intersection(df.index)

    aligned = [df.loc[common_idx] for df in dataframes]
    valid_mask = pd.Series(True, index=common_idx)
    for df in aligned:
        if not df.dtypes.apply(lambda t: str(t).startswith('bool')).all():
            valid_mask &= df.notna().all(axis=1)

    return tuple(df.loc[valid_mask] for df in aligned)

bars_per_day = 390 

# Sharpe compares return with the risk taken.
# Daily returns are used when there are enough bars.
def sharpe(portfolio, bars_per_day: int = bars_per_day) -> float:
    equity = portfolio.value()
    if isinstance(equity, pd.Series):
        equity = equity.to_frame()

    n_bars = len(equity)
    n_days = n_bars // bars_per_day

    if n_days < 2:
        returns       = equity.pct_change().dropna()
        per_bar       = returns.mean() / returns.std().replace(0, np.nan)
        annual_sharpe = per_bar * np.sqrt(bars_per_day * 252)
        return float(np.nanmean(annual_sharpe))
    
    day_end_idx   = [(d + 1) * bars_per_day - 1 for d in range(n_days)
                     if (d + 1) * bars_per_day - 1 < n_bars]
    daily_equity  = equity.iloc[day_end_idx]
    daily_returns = daily_equity.pct_change().dropna()

    if len(daily_returns) < 2:
        return float('nan')

    per_day_sharpe = daily_returns.mean() / daily_returns.std().replace(0, np.nan)
    annual_sharpe  = per_day_sharpe * np.sqrt(252) 
    return float(np.nanmean(annual_sharpe))

fixed_size = 10     
init_cash  = 100000    
size_type  = 'amount'  

L_range= list(range(10,45,5))
H_range= list(range(50,95,5))
W_range= list(range(10,90,5))
total_combination = len(L_range) * len(H_range) * len(W_range)


# This tries every L, H, and W combination.
# Each run stores the main backtest numbers for comparison.
def run_sweep(fee: float, label: str) -> pd.DataFrame:
    sweep_results = []
    sweep_skipped = 0

    for i, (L, H, W) in enumerate(itertools.product(L_range, H_range, W_range)):

        if L >= H:       
            sweep_skipped += 1
            continue

        entries, exits = generate_signals(prices, rsi, volatility, L, H, W)
        p, en, ex        = align_data(prices, entries, exits)

        if len(p) < 100:    
            sweep_skipped += 1
            continue

        try:
            pf = vbt.Portfolio.from_signals(
                p, en, ex,
                fees      = fee,
                init_cash = init_cash,
                size      = fixed_size,
                size_type = size_type,  
                freq      = '1T',
            )

            sharpe_val     = sharpe(pf, bars_per_day)
            if not np.isfinite(sharpe_val):
                sweep_skipped += 1
                continue

            total_return = float(np.nanmean(pf.total_return()))
            max_dd       = float(np.nanmean(pf.max_drawdown()))

            try:
                win_rate = float(np.nanmean(pf.trades.win_rate()))
                n_trades = int(pf.stats()["Total Trades"])
            except Exception:
                win_rate = float('nan')
                n_trades = 0

            sweep_results.append({
                "L": L, "H": H, "W": W,
                "Sharpe":       sharpe_val,
                "Total_Return": total_return,
                "Max_Drawdown": max_dd,
                "Win_Rate":     win_rate,
                "N_Trades":     n_trades,
            })

        except Exception:
            sweep_skipped += 1
            continue

        if (i + 1) % 30 == 0:
            print(f"{i+1}/{total_combination} are completed ")
                 
    df_out = pd.DataFrame(sweep_results)
    return df_out


# This picks a stable best parameter set.
# It prefers a good area of Sharpe values, not one lucky cell.
def parameters(results_df: pd.DataFrame, label: str) -> tuple:

    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sharpe"])

    if results_df.empty:
        raise ValueError(f"results_df is empty")
    
    top_q  = results_df["Sharpe"].quantile(0.75)
    top_df = results_df[results_df["Sharpe"] >= top_q].copy()

    # This checks nearby values around one result.
    # A good parameter should have decent neighbours too.
    def neighbour_sharpe(row, dL=5, dH=5, dW=10):
        nbrs = results_df[
            (results_df["L"].between(row["L"] - dL, row["L"] + dL)) &
            (results_df["H"].between(row["H"] - dH, row["H"] + dH)) &
            (results_df["W"].between(row["W"] - dW, row["W"] + dW))
        ]
        return nbrs["Sharpe"].mean() if len(nbrs) > 1 else row["Sharpe"]

    top_df = top_df.copy()
    top_df["Neighbour_Sharpe"] = top_df.apply(neighbour_sharpe, axis=1)
    best   = top_df.loc[top_df["Neighbour_Sharpe"].idxmax()].copy()
    plateau_cutoff = max(top_q, best["Sharpe"] - 0.10 * abs(best["Sharpe"]))
    plateau_df = results_df[results_df["Sharpe"] >= plateau_cutoff].copy()

    best["Plateau_Cutoff"] = plateau_cutoff
    best["Plateau_Count"]  = len(plateau_df)
    best["Plateau_L_Min"]  = int(plateau_df["L"].min())
    best["Plateau_L_Max"]  = int(plateau_df["L"].max())
    best["Plateau_H_Min"]  = int(plateau_df["H"].min())
    best["Plateau_H_Max"]  = int(plateau_df["H"].max())
    best["Plateau_W_Min"]  = int(plateau_df["W"].min())
    best["Plateau_W_Max"]  = int(plateau_df["W"].max())

    print(f"\n  ► {label} Best Parameters:")
    print(f"    L={int(best['L'])}, H={int(best['H'])}, W={int(best['W'])}")
    print(f"    Sharpe={best['Sharpe']:.3f} ")
    print(f"    Plateau Sharpe >= {best['Plateau_Cutoff']:.3f}")
    print(
        f"    Plateau Box: L={best['Plateau_L_Min']}-{best['Plateau_L_Max']}, "
        f"H={best['Plateau_H_Min']}-{best['Plateau_H_Max']}, "
        f"W={best['Plateau_W_Min']}-{best['Plateau_W_Max']} "
        f"({best['Plateau_Count']} points)"
    )

    return int(best["L"]), int(best["H"]), int(best["W"]), best

results_nf = run_sweep(fee=0.0,  label="(fees=0.0)")
results_f  = run_sweep(fee=0.01, label="(fees=0.01)")

print(f"\n[4] Selecting Best Parameters from Each Sweepe:-")
print(f"\n  Top 10 — No-Fee Sweep:")
print(results_nf.nlargest(10, "Sharpe").to_string(index=False))
print(f"\n  Top 10 — Fee Sweep (fees=0.01):")
print(results_f.nlargest(10, "Sharpe").to_string(index=False))

best_L_nf, best_H_nf, best_W_nf, row_nf = parameters(
    results_nf, "No-Fee Sweep"
)

best_L_f,  best_H_f,  best_W_f,  row_f  = parameters(
    results_f,  "Fee Sweep (0.01)"
)

print(f"\n[5] Final Backtest A - No-Fee parameters (L={best_L_nf}, H={best_H_nf}, W={best_W_nf})")

en_nf, ex_nf            = generate_signals(prices, rsi, volatility, best_L_nf, best_H_nf, best_W_nf)
p_nf, en_nf, ex_nf      = align_data(prices, en_nf, ex_nf)

pf_nofee = vbt.Portfolio.from_signals(
    p_nf, en_nf, ex_nf,
    fees=0.0, init_cash=init_cash,
    size=fixed_size, size_type=size_type, freq='1T',
)
sharpe_nofee = sharpe(pf_nofee, bars_per_day)

print(f"\nStats (No-Fee parameters, No-Fee backtest)")
print(pf_nofee.stats().to_string())
print(f"Annual Sharpe: {sharpe_nofee:.4f}")

print(f"\n[6] Final Backtest B - Fee parameters (L={best_L_f}, H={best_H_f}, W={best_W_f})")

en_f2, ex_f2          = generate_signals(prices, rsi, volatility, best_L_f, best_H_f, best_W_f)
p_f2, en_f2, ex_f2    = align_data(prices, en_f2, ex_f2)

pf_fee = vbt.Portfolio.from_signals(
    p_f2, en_f2, ex_f2,
    fees=0.01, init_cash=init_cash,
    size=fixed_size, size_type=size_type, freq='1T',
)
sharpe_fee = sharpe(pf_fee, bars_per_day)

print(f"\n Stats (Fee params, Fee=0.01 backtest)")
print(pf_fee.stats().to_string())
print(f"Annual Sharpe: {sharpe_fee:.4f}")

# Some vectorbt stats can fail when there are few trades.
# This returns N/A instead of stopping the full run.
def safe_stat(pf, key, fmt=".2%"):
    try:
        val = pf.stats()[key]
        return f"{val:{fmt}}" if '%' in fmt else f"{val:{fmt}}"
    except Exception:
        return "N/A"

# Win rate is useful, but it may not exist for every case.
# This keeps the final table clean even then.
def safe_winrate(pf):
    try:
        return f"{np.nanmean(pf.trades.win_rate()):.2%}"
    except Exception:
        return "N/A"

# This reads the total number of completed trades.
# If vectorbt cannot give it, the script keeps going.
def safe_trades(pf):
    try:
        return str(int(pf.stats()["Total Trades"]))
    except Exception:
        return "N/A"

# This reads total fees from the portfolio stats.
# It is mainly needed for the fee backtest result.
def safe_fees(pf):
    try:
        return f"₹{pf.stats()['Total Fees Paid']:.2f}"
    except Exception:
        return "N/A"

# This makes the plateau range easy to print.
# L, H, and W stay together in one table cell.
def plateau_text(row):
    return (
        f"L={int(row['Plateau_L_Min'])}-{int(row['Plateau_L_Max'])}, "
        f"H={int(row['Plateau_H_Min'])}-{int(row['Plateau_H_Max'])}, "
        f"W={int(row['Plateau_W_Min'])}-{int(row['Plateau_W_Max'])}"
    )

n_synth_nf = len(p_nf) // bars_per_day 
n_synth_f  = len(p_f2) // bars_per_day 

print("Final Comparison:-")

comparison = pd.DataFrame({
    "Metric": [
        "Parameters (L, H, W)",
        "Fee rate",
        "Annual Sharpe",
        "Total Return",
        "Max Drawdown",
        "Win Rate",
        "Total Trades",
        "Total Fees Paid",
        "Sharpe Plateau Box",
        "Synthetic days used",
    ],
    "No-Fee Strategy": [
        f"L={best_L_nf}, H={best_H_nf}, W={best_W_nf}",
        "0.0%",
        f"{sharpe_nofee:.3f}",
        f"{np.nanmean(pf_nofee.total_return()):.2%}",
        f"{np.nanmean(pf_nofee.max_drawdown()):.2%}",
        safe_winrate(pf_nofee),
        safe_trades(pf_nofee),
        "₹0.00",
        plateau_text(row_nf),
        str(n_synth_nf),
    ],
    "Fee Strategy (0.01)": [
        f"L={best_L_f},  H={best_H_f},  W={best_W_f}",
        "1.0%",
        f"{sharpe_fee:.3f}",
        f"{np.nanmean(pf_fee.total_return()):.2%}",
        f"{np.nanmean(pf_fee.max_drawdown()):.2%}",
        safe_winrate(pf_fee),
        safe_trades(pf_fee),
        safe_fees(pf_fee),
        plateau_text(row_f),
        str(n_synth_f),
    ],
})
print(comparison.to_string(index=False))

print(f"""
    Data note: Only ~{max(n_synth_nf, n_synth_f)} synthetic trading days available.
    500+ days needed for statistically reliable Sharpe estimation.
    Results demonstrate correct methodology, not real-world viability.
""")

# This draws the dark box around high-Sharpe cells.
# The yellow box marks the exact chosen parameter cell.
def add_plateau_box(data, ax, plateau_cutoff, best_x, best_y):
    plateau = data.where(data >= plateau_cutoff).dropna(how="all").dropna(axis=1, how="all")

    if not plateau.empty:
        rows = [list(data.index).index(x) for x in plateau.index]
        cols = [list(data.columns).index(x) for x in plateau.columns]
        x0, x1 = min(cols), max(cols)
        y0, y1 = min(rows), max(rows)

        ax.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0 + 1, y1 - y0 + 1,
            fill=False, edgecolor="#023047", linewidth=3.2
        ))
        ax.text(x0 + 0.05, y0 + 0.25, "Plateau",
                color="#023047", fontsize=9, fontweight="bold")

    if best_x in data.columns and best_y in data.index:
        best_col = list(data.columns).index(best_x)
        best_row = list(data.index).index(best_y)
        ax.add_patch(plt.Rectangle(
            (best_col, best_row), 1, 1,
            fill=False, edgecolor="#ffb703", linewidth=3.2
        ))

# This draws one clean heatmap from a pivot table.
# It can also mark the plateau and selected best point.
def plot_heatmap(data, ax, title, xlabel, ylabel, plateau_cutoff=None, best_x=None, best_y=None):
    if data.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\nNo data available")
        return

    sns.heatmap(data, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                cbar_kws={"label": "Annual Sharpe"})

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if plateau_cutoff is not None:
        add_plateau_box(data, ax, plateau_cutoff, best_x, best_y)

# This shows L and H for every W value.
# It is the full parameter map, so no W slice is hidden.
def plot_all_w_heatmaps(results_df, best_row, title, filename):
    n_cols = 4
    n_rows = int(np.ceil(len(W_range) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.2 * n_rows), sharex=True, sharey=True)
    vmin = results_df["Sharpe"].min()
    vmax = results_df["Sharpe"].max()

    for ax, W in zip(axes.flat, W_range):
        heat_data = results_df[results_df["W"] == W].pivot_table(
            index="L", columns="H", values="Sharpe"
        ).reindex(index=L_range, columns=H_range)

        sns.heatmap(
            heat_data, ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=vmin, vmax=vmax,
            linewidths=0.5, cbar=False, annot_kws={"size": 8}
        )

        best_x = int(best_row["H"]) if int(best_row["W"]) == W else None
        best_y = int(best_row["L"]) if int(best_row["W"]) == W else None
        add_plateau_box(heat_data, ax, best_row["Plateau_Cutoff"], best_x, best_y)

        ax.set_title(f"W = {W}")
        ax.set_xlabel("H")
        ax.set_ylabel("L")

    for ax in axes.flat[len(W_range):]:
        ax.set_axis_off()

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.subplots_adjust(right=0.90, top=0.91, hspace=0.35, wspace=0.12)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Annual Sharpe")

    plt.savefig(filename, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Saved {filename}")

# This shows how Sharpe values are spread out.
# A group of good values is better than one isolated spike.
def plot_sharpe_distribution(results_nf, results_f, row_nf, row_f):
    plot_df = pd.concat([
        results_nf.assign(Sweep="No-Fee Sweep"),
        results_f.assign(Sweep="Fee Sweep (0.01)")
    ], ignore_index=True)
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Sharpe"])

    palette = {
        "No-Fee Sweep": "#0077b6",
        "Fee Sweep (0.01)": "#d62828"
    }

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.histplot(
        data=plot_df, x="Sharpe", hue="Sweep", bins=24,
        kde=True, stat="density", common_norm=False,
        element="step", fill=True, alpha=0.28,
        palette=palette, ax=ax
    )

    for best_row, label, color in [
        (row_nf, "Best No-Fee", palette["No-Fee Sweep"]),
        (row_f, "Best Fee", palette["Fee Sweep (0.01)"])
    ]:
        ax.axvline(best_row["Sharpe"], color=color, linestyle="--", linewidth=2.2)
        ax.text(
            best_row["Sharpe"], ax.get_ylim()[1] * 0.90,
            f"{label}: {best_row['Sharpe']:.2f}",
            rotation=90, va="top", ha="right", color=color, fontweight="bold"
        )

    ax.set_title("Sharpe Distribution Across Parameter Sweeps", fontsize=15, fontweight="bold")
    ax.set_xlabel("Annualised Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Sharpe_Distribution.png", dpi=180, bbox_inches="tight")
    plt.savefig("Plot_2_Sharpe_Distribution.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved Sharpe_Distribution.png")

# This shows RSI and volatility for every stock.
# It helps check if the signals make sense before the backtest.
def plot_rsi_volatility(prices, rsi, volatility, W):
    stock_names = list(prices.columns)
    colors = ["#0077b6", "#2a9d8f", "#e9c46a", "#f4a261", "#d62828"]
    vol_ma = volatility.rolling(window=W, min_periods=W).mean()

    fig, axes = plt.subplots(len(stock_names), 2, figsize=(18, 3.2 * len(stock_names)), sharex=True)
    if len(stock_names) == 1:
        axes = np.array([axes])

    x = prices.index.to_numpy()
    fig.suptitle(
        f"RSI-14 and Volatility Filter View Across {len(stock_names)} Stocks",
        fontsize=16, fontweight="bold"
    )

    for idx, stock in enumerate(stock_names):
        color      = colors[idx % len(colors)]
        rsi_values = rsi[stock].astype(float)
        vol_values = volatility[stock].astype(float)
        ma_values  = vol_ma[stock].astype(float)

        axes[idx][0].plot(x, rsi_values.to_numpy(), color=color, linewidth=1.4)
        axes[idx][0].fill_between(
            x, 70, rsi_values.to_numpy(),
            where=(rsi_values >= 70).fillna(False).to_numpy(),
            color="#d62828", alpha=0.15
        )
        axes[idx][0].fill_between(
            x, rsi_values.to_numpy(), 30,
            where=(rsi_values <= 30).fillna(False).to_numpy(),
            color="#2a9d8f", alpha=0.15
        )
        axes[idx][0].axhline(70, color="#d62828", linestyle="--", linewidth=1)
        axes[idx][0].axhline(30, color="#2a9d8f", linestyle="--", linewidth=1)
        axes[idx][0].set_ylim(0, 100)
        axes[idx][0].set_title(f"{stock} RSI-14")
        axes[idx][0].set_ylabel("RSI")
        axes[idx][0].grid(True, alpha=0.3)

        axes[idx][1].plot(x, vol_values.to_numpy(), color=color, linewidth=1.35, label="Volatility")
        axes[idx][1].plot(x, ma_values.to_numpy(), color="#111827", linestyle="--", linewidth=1.1,
                          label=f"Vol MA W={W}")
        axes[idx][1].fill_between(x, 0, vol_values.to_numpy(), color=color, alpha=0.12)
        axes[idx][1].set_title(f"{stock} Rolling Volatility")
        axes[idx][1].set_ylabel("Std. Dev.")
        axes[idx][1].set_ylim(bottom=0)
        axes[idx][1].grid(True, alpha=0.3)
        axes[idx][1].legend(frameon=False, loc="upper right")

    axes[-1][0].set_xlabel("Minute Bar")
    axes[-1][1].set_xlabel("Minute Bar")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("RSI_Volatility_Stocks.png", dpi=180, bbox_inches="tight")
    plt.savefig("Plot_1_RSI_Volatility.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved RSI_Volatility_Stocks.png")

plot_sharpe_distribution(results_nf, results_f, row_nf, row_f)
plot_rsi_volatility(prices, rsi, volatility, best_W_nf)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "Parameter Sensitivity - Honest Annualised Sharpe\n"
    "Left: No-Fee Sweep in Right: Fee=0.01 Sweep\n"
    "(10 share/trade)",
    fontsize=12, fontweight="bold"
)

for col_idx, (res_df, best_L, best_H, best_W, best_row, lbl) in enumerate([
    (results_nf, best_L_nf, best_H_nf, best_W_nf, row_nf, f"No-Fee (L={best_L_nf},H={best_H_nf},W={best_W_nf})"),
    (results_f,  best_L_f,  best_H_f,  best_W_f,  row_f,  f"Fee=0.01 (L={best_L_f},H={best_H_f},W={best_W_f})"),
]):
    # Row 0: L vs H heatmap
    hm_LH = res_df[res_df["W"] == best_W].pivot_table(
        index="L", columns="H", values="Sharpe"
    ).reindex(index=L_range, columns=H_range)
    plot_heatmap(
        hm_LH, axes[0][col_idx],
        f"{lbl}\nL vs H  [W={best_W} fixed]",
        "H — Sell RSI threshold",
        "L — Buy RSI threshold",
        plateau_cutoff=best_row["Plateau_Cutoff"],
        best_x=best_H,
        best_y=best_L
    )

    # Row 1: H vs W heatmap
    hm_HW = res_df[res_df["L"] == best_L].pivot_table(
        index="H", columns="W", values="Sharpe"
    ).reindex(index=H_range, columns=W_range)
    plot_heatmap(
        hm_HW, axes[1][col_idx],
        f"H vs W  [L={best_L} fixed]",
        "W — Volatility window",
        "H — Sell RSI threshold",
        plateau_cutoff=best_row["Plateau_Cutoff"],
        best_x=best_W,
        best_y=best_H
    )

plt.tight_layout()
plt.savefig("parameter_heatmaps_dual.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved parameter_heatmaps_dual.png")

plot_all_w_heatmaps(
    results_nf, row_nf,
    "No-Fee Sweep - L vs H for Every W",
    "all_w_heatmaps_nofee.png"
)
plot_all_w_heatmaps(
    results_f, row_f,
    "Fee Sweep (0.01) - L vs H for Every W",
    "all_w_heatmaps_fee.png"
)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.suptitle(
    "Equity Curves - Each Strategy with Its Own Optimal Parameters\n"
    f"Left: No-Fee (L={best_L_nf},H={best_H_nf},W={best_W_nf})  |  "
    f"Right: Fee=0.01 (L={best_L_f},H={best_H_f},W={best_W_f})",
    fontsize=12, fontweight="bold"
)

pf_nofee.value().plot(ax=axes[0], linewidth=1.3)
axes[0].axhline(init_cash, color="gray", linestyle="--", linewidth=0.8,
                label="Starting capital")
axes[0].set_title(
    f"Strategy A - No-Fee Parameters, Annual Sharpe = {sharpe_nofee:.3f}"
)
axes[0].set_ylabel("Portfolio Value (₹)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

pf_fee.value().plot(ax=axes[1], linewidth=1.3)
axes[1].axhline(init_cash, color="gray", linestyle="--", linewidth=0.8,
                label="Starting capital")
axes[1].set_title(
    f"Strategy B - Fee=0.01 Parameters, Annual Sharpe = {sharpe_fee:.3f}"
)
axes[1].set_ylabel("Portfolio Value (₹)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("equity_curves_dual.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved equity_curves_dual.png")

print(f"""
  Strategy   : Mean Reversion (RSI-14 + Volatility Filter)
  Universe   : {len(prices.columns)} stocks | 1-min bars | integer index
  Capital    : ₹{init_cash:,} | {fixed_size} share per trade (size_type='{size_type}')

  ┌─────────────────────────────────────────────────────────┐
  │              NO-FEE STRATEGY                            │
  │  Parameters : L={best_L_nf}, H={best_H_nf}, W={best_W_nf}                       
  │  Optimised on : fees = 0.0                              │
  │  Annual Sharpe : {sharpe_nofee:.4f}                     |
  ├─────────────────────────────────────────────────────────┤
  │              FEE STRATEGY (0.01 per trade)              │
  │  Parameters : L={best_L_f},  H={best_H_f},  W={best_W_f}|                      
  │  Optimised on : fees = 0.01                             │
  │  Annual Sharpe : {sharpe_fee:.4f}                       |
  └─────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ## Why the Selected Parameters Are Robust
#
# The selected parameters are not picked only because one single combination gives the highest Sharpe ratio. The code also checks nearby values of `L`, `H`, and `W`, and prefers a region where Sharpe stays high around the chosen point.
#
# This is important because 1-minute data is noisy, and one isolated best value can happen by chance. A plateau means the strategy still performs well when the thresholds are moved slightly, so the result is less likely to be overfitted.
#
# In the heatmaps, the dark blue box shows the stable Sharpe plateau and the yellow box shows the final selected parameters. Since the selected point sits inside a wider good area, it is more robust than just chasing the single highest cell.
