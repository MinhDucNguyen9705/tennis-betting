#!/bin/bash
# =========================
# Quick Start: Train Model & Run Backtest Demo
# =========================

echo "=================================="
echo "Tennis Betting - Train & Backtest"
echo "=================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Activating virtual environment..."
    source venv/bin/activate
fi

# Step 1: Train model and run backtest
echo "📊 Step 1: Training model and running backtest..."
echo ""
python3 train_and_backtest.py --config train_and_backtest.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training and backtesting completed!"
    echo ""
    echo "📈 Step 2: Starting interactive demo dashboard..."
    echo ""
    echo "Dashboard will open at: http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    echo ""
    sleep 2
    
    cd backtest
    python3 app.py
else
    echo ""
    echo "✗ Training/backtesting failed. Check errors above."
    exit 1
fi
