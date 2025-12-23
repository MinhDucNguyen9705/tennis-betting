#!/usr/bin/env python3
# =========================
# Integrated Train & Backtest Pipeline
# =========================

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtest'))

from models.run import train, evaluate, load_config as load_model_config, setup_logging
from backtest.backtest_utils import TwoSidedKellyBacktester, TopPlayerKellyBacktester, TwoSidedBacktester


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_inference_for_backtest(model_path: str, csv_files: list, config: dict, logger) -> pd.DataFrame:
    """
    Run inference on CSV files with odds for backtesting.
    
    Args:
        model_path: Path to trained model
        csv_files: List of CSV file paths
        config: Configuration dict
        logger: Logger instance
        
    Returns:
        Combined DataFrame with predictions and odds
    """
    from models.inference import load_model, prepare_for_inference
    
    logger.info("Loading trained model for inference...")
    bundle = load_model(model_path)
    model = bundle['model']
    features = bundle['features']
    threshold = bundle.get('threshold', 0.5)
    
    all_results = []
    
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file}...")
        
        # Load CSV with odds
        raw_df = pd.read_csv(csv_file, sep=config['data']['csv_sep'])
        logger.info(f"  Loaded {len(raw_df)} matches")
        
        # Prepare for inference
        prepared = prepare_for_inference(raw_df)
        
        # Check for missing features
        missing = [c for c in features if c not in prepared.columns]
        if missing:
            logger.warning(f"  Missing features: {missing}")
            continue
        
        # Predict
        X = prepared[features].fillna(-1).values
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        
        # Create result dataframe
        result = pd.DataFrame()
        
        # Keep essential columns
        keep_cols = ['tournament_date', 'tournament', 'Name_1', 'Name_2', 
                     'Winner', 'PS_1', 'PS_2', 'B365_1', 'B365_2',
                     'Ranking_1', 'Ranking_2']
        
        for col in keep_cols:
            if col in raw_df.columns:
                result[col] = raw_df[col]
        
        # Add predictions
        result['Prob_P1'] = proba
        result['pred_player1_win'] = pred
        
        # Transform Winner for backtesting
        if 'Winner' in result.columns:
            result['Victory'] = (result['Winner'] == 0).astype(int)
        
        all_results.append(result)
    
    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"Total matches for backtesting: {len(combined)}")
    
    return combined


def run_backtest_strategies(df: pd.DataFrame, config: dict, logger):
    """
    Run all enabled backtest strategies.
    
    Args:
        df: DataFrame with predictions and odds
        config: Configuration dict
        logger: Logger instance
        
    Returns:
        Dictionary of backtest results
    """
    results = {}
    initial_capital = config['backtest']['initial_capital']
    
    logger.info("\n" + "="*60)
    logger.info("BACKTESTING PHASE")
    logger.info("="*60)
    
    # Kelly Strategy
    if config['backtest']['strategies']['kelly']['enabled']:
        logger.info("\n--- Running Two-Sided Kelly Strategy ---")
        params = config['backtest']['strategies']['kelly']
        bt = TwoSidedKellyBacktester(
            initial_capital=initial_capital,
            kelly_multiplier=params['kelly_multiplier'],
            max_stake_pct=params['max_stake_pct']
        )
        bt.run(df.copy(), prob_col='Prob_P1', p1_col='Name_1', winner_col='Victory')
        results['kelly'] = bt
    
    # Top Player Kelly Strategy
    if config['backtest']['strategies']['top_player_kelly']['enabled']:
        logger.info("\n--- Running Top Player Kelly Strategy ---")
        params = config['backtest']['strategies']['top_player_kelly']
        bt = TopPlayerKellyBacktester(
            top_n=params['top_n'],
            initial_capital=initial_capital,
            kelly_multiplier=params['kelly_multiplier'],
            max_stake_pct=params['max_stake_pct']
        )
        bt.run(df.copy(), prob_col='Prob_P1', p1_col='Name_1',
               rank1_col='Ranking_1', rank2_col='Ranking_2', winner_col='Victory')
        results['top_player_kelly'] = bt
    
    # Simple Fixed Strategy
    if config['backtest']['strategies']['simple_fixed']['enabled']:
        logger.info("\n--- Running Simple Fixed Bet Strategy ---")
        params = config['backtest']['strategies']['simple_fixed']
        bt = TwoSidedBacktester(
            initial_capital=initial_capital,
            bet_amount=params['bet_amount'],
            threshold=params['threshold']
        )
        bt.run(df.copy(), prob_col='Prob_P1', p1_col='Name_1', winner_col='Victory')
        results['simple_fixed'] = bt
    
    return results


def plot_comparison(results: dict, config: dict, logger):
    """
    Plot comparison of all strategies.
    
    Args:
        results: Dictionary of backtest results
        config: Configuration dict
        logger: Logger instance
    """
    plt.figure(figsize=(12, 7))
    
    colors = {
        'kelly': '#27ae60',
        'top_player_kelly': '#3498db',
        'simple_fixed': '#e74c3c'
    }
    
    labels = {
        'kelly': 'Two-Sided Kelly',
        'top_player_kelly': 'Top Player Kelly',
        'simple_fixed': 'Simple Fixed Bet'
    }
    
    initial_capital = config['backtest']['initial_capital']
    
    for name, bt in results.items():
        plt.plot(bt.capital_history, 
                color=colors.get(name, 'gray'),
                label=labels.get(name, name),
                linewidth=2)
    
    plt.axhline(y=initial_capital, color='red', linestyle='--', 
                label='Initial Capital', linewidth=1)
    
    plt.title('Backtest Strategy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Bets', fontsize=12)
    plt.ylabel('Capital ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = config['output']['equity_plot']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info(f"Equity curve saved to {output_path}")
    
    if config['backtest']['plot_results']:
        plt.show()


def generate_backtest_report(results: dict, config: dict, predictions_df: pd.DataFrame, logger):
    """
    Generate comprehensive backtest report.
    
    Args:
        results: Dictionary of backtest results
        config: Configuration dict
        predictions_df: DataFrame with predictions
        logger: Logger instance
    """
    report_path = config['output']['backtest_report']
    initial_capital = config['backtest']['initial_capital']
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TENNIS BETTING - INTEGRATED BACKTEST REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Training Years: {config['data']['train_years']}\n")
        f.write(f"Calibration Years: {config['data']['calib_years']}\n")
        f.write(f"Backtest Years: {config['data']['backtest_years']}\n")
        f.write(f"Initial Capital: ${initial_capital:,.2f}\n")
        f.write(f"Total Matches: {len(predictions_df)}\n\n")
        
        f.write("STRATEGY RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for name, bt in results.items():
            profit = bt.current_capital - initial_capital
            roi = (profit / initial_capital) * 100
            
            # Calculate max drawdown
            peak = initial_capital
            max_dd = 0
            for x in bt.capital_history:
                if x > peak:
                    peak = x
                dd = (peak - x) / peak
                if dd > max_dd:
                    max_dd = dd
            
            f.write(f"{name.upper().replace('_', ' ')}\n")
            f.write("-"*60 + "\n")
            f.write(f"Final Capital: ${bt.current_capital:,.2f}\n")
            f.write(f"Profit/Loss: ${profit:,.2f}\n")
            f.write(f"ROI: {roi:.2f}%\n")
            f.write(f"Max Drawdown: {max_dd*100:.2f}%\n")
            f.write(f"Total Bets: {len(bt.capital_history)-1}\n\n")
    
    logger.info(f"Backtest report saved to {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Integrated Train & Backtest Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and backtest
  python train_and_backtest.py --config train_and_backtest.yaml
  
  # Only backtest (requires trained model)
  python train_and_backtest.py --config train_and_backtest.yaml --skip-train
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='train_and_backtest.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training and use existing model'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config['output']['log_level'])
    
    logger.info("="*60)
    logger.info("INTEGRATED TRAIN & BACKTEST PIPELINE")
    logger.info("="*60)
    
    try:
        # PHASE 1: TRAIN MODEL (if not skipped)
        if not args.skip_train:
            logger.info("\n📊 PHASE 1: TRAINING MODEL")
            
            # Convert to models/run.py compatible config
            model_config = {
                'data': config['data'],
                'model': config['model'],
                'training': config['training'],
                'features': config['features'],
                'output': config['output']
            }
            
            train(model_config, logger)
            logger.info("✓ Model training completed")
        else:
            logger.info("\n⏭️  Skipping training, using existing model")
        
        # PHASE 2: RUN INFERENCE FOR BACKTEST
        logger.info("\n🔮 PHASE 2: GENERATING PREDICTIONS FOR BACKTEST")
        
        # Find CSV files with odds for backtest years
        csv_files = []
        for year in config['data']['backtest_years']:
            csv_path = os.path.join(
                config['data']['csv_dir'],
                f"matches_data_{year}.csv"
            )
            if os.path.exists(csv_path):
                csv_files.append(csv_path)
            else:
                logger.warning(f"CSV file not found: {csv_path}")
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found for backtesting")
        
        predictions_df = run_inference_for_backtest(
            config['output']['model_path'],
            csv_files,
            config,
            logger
        )
        
        # Save predictions
        predictions_df.to_csv(config['output']['predictions_path'], index=False)
        logger.info(f"✓ Predictions saved to {config['output']['predictions_path']}")
        
        # PHASE 3: RUN BACKTESTS
        logger.info("\n💰 PHASE 3: BACKTESTING STRATEGIES")
        
        results = run_backtest_strategies(predictions_df, config, logger)
        
        # PHASE 4: GENERATE RESULTS
        logger.info("\n📈 PHASE 4: GENERATING RESULTS")
        
        plot_comparison(results, config, logger)
        generate_backtest_report(results, config, predictions_df, logger)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("\nResults:")
        logger.info(f"  - Model: {config['output']['model_path']}")
        logger.info(f"  - Predictions: {config['output']['predictions_path']}")
        logger.info(f"  - Report: {config['output']['backtest_report']}")
        logger.info(f"  - Equity Plot: {config['output']['equity_plot']}")
        
    except Exception as e:
        logger.error(f"\n✗ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
