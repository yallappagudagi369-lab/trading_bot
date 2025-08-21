import os
import logging
import math
import pandas as pd
import requests
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ==== CONFIG ====
TOKEN = os.getenv("TOKEN")            # set in Railway → Variables
CHAT_ID = os.getenv("CHAT_ID", "")    # optional; not required for this bot
CRYPTO_TIMEFRAME = "5m"
FOREX_GOLD_TIMEFRAME = "5m"
STOCK_TIMEFRAME = "1d"
ALLOWED_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","4h","1d"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ema-atr-signal-bot")

# ==== UTILS ====
def normalize_for_yf(symbol: str) -> str:
    s = symbol.upper()
    if s in ("XAUUSD", "GOLD"):
        return "GC=F"
    if s.endswith("USDT") and "-" not in s:
        base = s.replace("USDT", "USD")
        return base[:3] + "-" + base[3:]
    return s

def fetch_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame | None:
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","tbbav","tbqav","ignore"]
        df = pd.DataFrame(data, columns=cols)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.rename(columns=str.capitalize, inplace=True)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        return df if not df.empty else None
    except Exception as e:
        log.warning(f"Binance fetch failed for {symbol}: {e}")
        return None

def fetch_yf(symbol: str, interval: str, lookback: str="90d") -> pd.DataFrame | None:
    try:
        yf_symbol = normalize_for_yf(symbol)
        df = yf.download(yf_symbol, period=lookback, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        out = df[["Open","High","Low","Close","Volume"]].dropna()
        return out if not out.empty else None
    except Exception as e:
        log.warning(f"yfinance fetch failed for {symbol}: {e}")
        return None

def fetch_data(symbol: str) -> pd.DataFrame | None:
    s = symbol.upper()
    # Crypto
    if s.endswith(("USDT","BUSD","BTC","ETH")):
        df = fetch_binance(s, CRYPTO_TIMEFRAME)
        return df if df is not None else fetch_yf(s, CRYPTO_TIMEFRAME)
    # Forex/Gold
    elif s in ("XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD","USDCHF"):
        for interval in ["5m","15m","1h"]:
            df = fetch_yf(s, interval)
            if df is not None and not df.empty:
                return df
    # Stocks
    else:
        return fetch_yf(s, STOCK_TIMEFRAME)

# ==== INDICATORS ====
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),(high - prev_close).abs(),(low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ==== SIGNALS ====
def decide_signal(df: pd.DataFrame) -> tuple[str,float]:
    if df is None or df.empty or len(df) < 20:
        return "⚠️ Not enough data", 0.0
    df = df.copy()
    close = df["Close"]
    df["EMA9"], df["EMA15"] = ema(close,9), ema(close,15)
    df["ATR14"] = atr(df,14)
    ema9, ema15, atr0 = df["EMA9"].iloc[-1], df["EMA15"].iloc[-1], df["ATR14"].iloc[-1]

    if float(ema9) > float(ema15):
        signal = "BUY 📈"
    elif float(ema9) < float(ema15):
        signal = "SELL 📉"
    else:
        signal = "⚪ NO TRADE (weak/unclear)"
    return signal, float(atr0)

# ==== TRADE LEVELS ====
def calculate_trade_levels(price: float, atr_value: float, signal: str, 
                           atr_mult_sl=1.5, atr_mult_tp=2.5,
                           min_sl_points=400, min_tp_points=1000):
    """
    Calculate entry, SL, and TP levels.
    Ensures minimum points distance but also adapts to market volatility (ATR).
    """
    entry = round(price, 5)

    if signal.startswith("BUY"):
        sl = price - max(atr_value * atr_mult_sl, min_sl_points)
        tp = price + max(atr_value * atr_mult_tp, min_tp_points)
    elif signal.startswith("SELL"):
        sl = price + max(atr_value * atr_mult_sl, min_sl_points)
        tp = price - max(atr_value * atr_mult_tp, min_tp_points)
    else:
        return entry, None, None

    return entry, round(sl, 5), round(tp, 5)

# ==== TELEGRAM BOT ====
HELP_TEXT = (
    "🤖 Ultimate EMA/ATR Signal Bot\n\n"
    "Crypto (Scalp) → 5m\n"
    "Forex/Gold (Scalp) → 5m/15m fallback\n"
    "Stocks (Swing) → 1D\n\n"
    "Use: /signal SYMBOL\n"
    "Examples:\n"
    "• /signal BTCUSDT\n"
    "• /signal XAUUSD\n"
    "• /signal EURUSD\n"
    "• /signal INFY.NS"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot online.\n\n" + HELP_TEXT)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text or ""
        symbol = text.partition(" ")[2].strip().upper()
        if not symbol:
            await update.message.reply_text("Usage: /signal SYMBOL\nExample: /signal BTCUSDT")
            return

        await update.message.reply_text(f"⏳ Fetching {symbol} ...")
        df = fetch_data(symbol)
        if df is None or df.empty:
            await update.message.reply_text(f"❌ No data for {symbol}.")
            return

        signal, atr_value = decide_signal(df)
        entry, sl, tp = calculate_trade_levels(df["Close"].iloc[-1], atr_value, signal)

        msg = (
            f"📊 {symbol}\n"
            f"Signal: *{signal}*\n"
            f"Entry: `{entry}`\n"
            f"Stop Loss: `{sl}`\n"
            f"Take Profit: `{tp}`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        logging.exception("Error in /signal handler")
        await update.message.reply_text(f"⚠️ Error: {e}")

# ==== MAIN ====
def main():
    if not TOKEN:
        raise SystemExit("Missing TOKEN. Set it in Railway → Variables.")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("signal", cmd_signal))
    log.info("🚀 EMA/ATR Signal Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
