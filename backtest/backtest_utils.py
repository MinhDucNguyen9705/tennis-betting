import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

class TwoSidedKellyBacktester:
    def __init__(self, initial_capital=1000, kelly_multiplier=0.5, max_stake_pct=0.15):
        """
        :param kelly_multiplier: Hệ số Kelly (0.5 là Half-Kelly, khuyên dùng).
        :param max_stake_pct: Giới hạn cược tối đa (VD: 0.15 là 15% vốn).
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.kelly_multiplier = kelly_multiplier
        self.max_stake_pct = max_stake_pct
        
        self.capital_history = [initial_capital]
        self.stakes_history = []
        self.metrics = {}
        self.trades = []

    def calculate_kelly_stake(self, prob, odds):
        """Tính % vốn cược theo công thức Kelly"""
        if odds <= 1: return 0.0
        
        b = odds - 1
        p = prob
        q = 1 - p
        
        # Công thức: f = (bp - q) / b
        f = (b * p - q) / b
        
        # Chỉ cược nếu f > 0
        if f <= 0: return 0.0
        
        # Điều chỉnh Fractional & Max Cap
        f_final = f * self.kelly_multiplier
        return min(f_final, self.max_stake_pct)

    def run(self, df, prob_col='Prob_P1', p1_col='Name_1', winner_col='Victory'):
        print(f"--- BẮT ĐẦU KELLY 2 CHIỀU (Multiplier: {self.kelly_multiplier}) ---")
        
        bets_won = 0
        bets_lost = 0
        skipped = 0
        
        data = df.copy()
        
        for idx, row in data.iterrows():
            if self.current_capital <= 0:
                print("!!! CHÁY TÀI KHOẢN !!!")
                break
                
            try:
                name_1 = row[p1_col]
                winner = row[winner_col]

                odds_p1 = row['PS_1']
                odds_p2 = row['PS_2']
                if winner == 0:
                    # odds_p1 = row['PSW']
                    # odds_p2 = row['PSL']
                    p1_outcome = 1 
                else:
                    # odds_p1 = row['PSL']
                    # odds_p2 = row['PSW']
                    p1_outcome = 0
                
                if pd.isna(odds_p1) or pd.isna(odds_p2): continue

                
                prob_p1 = row[prob_col]
                stake_p1 = self.calculate_kelly_stake(prob_p1, odds_p1)
                
                
                prob_p2 = 1.0 - prob_p1
                stake_p2 = self.calculate_kelly_stake(prob_p2, odds_p2)
                
                selected_bet = None
                final_stake_pct = 0.0
                
                # So sánh xem kèo nào ngon hơn
                if stake_p1 > 0 and stake_p1 >= stake_p2:
                    selected_bet = 'P1'
                    final_stake_pct = stake_p1
                elif stake_p2 > 0 and stake_p2 > stake_p1:
                    selected_bet = 'P2'
                    final_stake_pct = stake_p2
                
                # 5. XỬ LÝ KẾT QUẢ
                if selected_bet:
                    bet_amount = self.current_capital * final_stake_pct
                    
                    # Logic thắng thua
                    is_win = False
                    if selected_bet == 'P1' and p1_outcome == 1:
                        is_win = True
                        profit = bet_amount * (odds_p1 - 1)
                    elif selected_bet == 'P2' and p1_outcome == 0:
                        is_win = True
                        profit = bet_amount * (odds_p2 - 1)
                    
                    # Cập nhật tiền
                    if is_win:
                        self.current_capital += profit
                        bets_won += 1
                    else:
                        self.current_capital -= bet_amount
                        bets_lost += 1
                        
                    self.stakes_history.append(final_stake_pct)
                else:
                    skipped += 1
                    self.stakes_history.append(0)
                # print(selected_bet, final_stake_pct)
                odds = odds_p1 if selected_bet == "P1" else odds_p2
                prob = prob_p1 if selected_bet == "P1" else prob_p2
                payout = profit if is_win else -bet_amount
                # print(odds, prob, selected_bet, final_stake_pct)

                self.trades.append({
                    "date": row.get("tournament_date", None),
                    # "tournament_type": row.get("tournament_type", None),
                    "tournament_level": row.get("tournament_level", None),
                    "surface": row.get("tournament_surface", row.get("surface", None)),
                    "round": row.get("round", None),

                    "bet_side": selected_bet,
                    "prob": float(prob),
                    "odds": float(odds),
                    "stake": float(bet_amount),
                    "pnl": float(payout),
                    "is_win": int(is_win),

                    # optional rank analysis if columns exist
                    "rank1": row.get("Ranking_1", None),
                    "rank2": row.get("Ranking_2", None),
                    "rank_diff": (row.get("Ranking_2", None) - row.get("Ranking_1", None))
                                if (pd.notna(row.get("Ranking_1", None)) and pd.notna(row.get("Ranking_2", None)))
                                else None,
                })

            except Exception as e:
                continue
            
            if self.current_capital <= 0:
                print("!!! CHÁY TÀI KHOẢN !!!")
                break
            if self.current_capital == self.capital_history[-1]:
                continue
            self.capital_history.append(self.current_capital)
            
        self._generate_report(bets_won, bets_lost, skipped)

    def _generate_report(self, wins, losses, skipped):
        total_bets = wins + losses
        profit = self.current_capital - self.initial_capital
        roi = (profit / self.initial_capital * 100) # ROI trên vốn gốc
        
        peak = self.initial_capital
        max_dd = 0
        for x in self.capital_history:
            if x > peak: peak = x
            dd = (peak - x) / peak
            if dd > max_dd: max_dd = dd

        # print("\n--- KẾT QUẢ ---")
        # print(f"Vốn cuối: ${self.current_capital:.2f} (Lãi: ${profit:.2f})")
        # print(f"Tổng lệnh: {total_bets} (Win Rate: {wins/total_bets*100:.1f}%)")
        # print(f"ROI (trên vốn): {roi:.2f}%")
        # print(f"Max Drawdown: {max_dd*100:.2f}%")
        # print("---------------")

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.capital_history, color='purple', label='Bankroll')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--')
        plt.title(f'Two-Sided Kelly Growth (Max Stake: {self.max_stake_pct*100}%)')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

class TopPlayerKellyBacktester:
    def __init__(self, top_n=8, initial_capital=1000, kelly_multiplier=0.5, max_stake_pct=0.15):
        """
        :param top_n: Chỉ xét các trận đấu có ít nhất 1 tay vợt nằm trong Top N này.
        :param kelly_multiplier: Hệ số Kelly (0.5 = Half Kelly).
        :param max_stake_pct: Giới hạn cược tối đa (% vốn).
        """
        self.top_n = top_n
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.kelly_multiplier = kelly_multiplier
        self.max_stake_pct = max_stake_pct
        
        self.capital_history = [initial_capital]
        self.metrics = {}

    def calculate_kelly_stake(self, prob, odds):
        """Tính toán Kelly Stake (dùng chung logic chuẩn)"""
        if odds <= 1: return 0.0
        b = odds - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        if f <= 0: return 0.0
        f_final = f * self.kelly_multiplier
        return min(f_final, self.max_stake_pct)

    def run(self, df, prob_col='Prob_P1', p1_col='Name_1', 
            rank1_col='Ranking_1', rank2_col='Ranking_2', winner_col='Victory'):
        
        print(f"--- BẮT ĐẦU: TOP {self.top_n} STRATEGY (Kelly x{self.kelly_multiplier}) ---")
        
        bets_won = 0
        bets_lost = 0
        skipped_low_rank = 0 # Đếm số trận bị bỏ qua do không đủ chuẩn Top Player
        skipped_no_edge = 0  # Đếm số trận bỏ qua do không thơm
        
        data = df.copy()
        
        for idx, row in data.iterrows():
            if self.current_capital <= 0:
                print("!!! CHÁY TÀI KHOẢN !!!")
                break
            
            # =================================================
            # BƯỚC 1: LỌC RANKING (FILTER STRATEGY)
            # =================================================
            r1 = row[rank1_col]
            r2 = row[rank2_col]
            
            # Xử lý dữ liệu rank bị thiếu (NaN) -> Coi là rank rất thấp (9999)
            r1 = 9999 if pd.isna(r1) else r1
            r2 = 9999 if pd.isna(r2) else r2
            
            # Điều kiện: Ít nhất 1 trong 2 người phải nằm trong Top N
            # Ví dụ: Top 5 đấu với Top 100 -> VẪN LẤY (vì có Top 5)
            is_top_match = (r1 <= self.top_n) or (r2 <= self.top_n)
            
            if not is_top_match:
                skipped_low_rank += 1
                if self.current_capital == self.capital_history[-1]:
                    continue
                self.capital_history.append(self.current_capital) # Giữ nguyên tiền
                continue

            # =================================================
            # BƯỚC 2: LOGIC KELLY 2 CHIỀU (NHƯ CŨ)
            # =================================================
            name_1 = row[p1_col]
            winner = row[winner_col]
            
            # Map Odds
            odds_p1 = row['PS_1']
            odds_p2 = row['PS_2']
            if winner == 0:
                # odds_p1 = row['PSW']
                # odds_p2 = row['PSL']
                p1_outcome = 1 
            else:
                # odds_p1 = row['PSL']
                # odds_p2 = row['PSW']
                p1_outcome = 0
            
            if pd.isna(odds_p1) or pd.isna(odds_p2): 
                self.capital_history.append(self.current_capital)
                continue

            # Tính Kelly cho cả 2 cửa
            prob_p1 = row[prob_col]
            stake_p1 = self.calculate_kelly_stake(prob_p1, odds_p1)
            
            prob_p2 = 1.0 - prob_p1
            stake_p2 = self.calculate_kelly_stake(prob_p2, odds_p2)
            
            # Chọn kèo ngon nhất
            selected_bet = None
            final_stake_pct = 0.0
            
            if stake_p1 > 0 and stake_p1 >= stake_p2:
                selected_bet = 'P1'
                final_stake_pct = stake_p1
            elif stake_p2 > 0 and stake_p2 > stake_p1:
                selected_bet = 'P2'
                final_stake_pct = stake_p2
            
            # Vào lệnh
            if selected_bet:
                bet_amount = self.current_capital * final_stake_pct
                is_win = False
                
                if selected_bet == 'P1' and p1_outcome == 1:
                    is_win = True
                    profit = bet_amount * (odds_p1 - 1)
                elif selected_bet == 'P2' and p1_outcome == 0:
                    is_win = True
                    profit = bet_amount * (odds_p2 - 1)
                
                if is_win:
                    self.current_capital += profit
                    bets_won += 1
                else:
                    self.current_capital -= bet_amount
                    bets_lost += 1
            else:
                skipped_no_edge += 1

            if self.current_capital == self.capital_history[-1]:
                continue
            self.capital_history.append(self.current_capital)
            
        # TỔNG KẾT
        self._generate_report(bets_won, bets_lost, skipped_low_rank, skipped_no_edge)

    def _generate_report(self, wins, losses, skip_rank, skip_edge):
        total_bets = wins + losses
        profit = self.current_capital - self.initial_capital
        roi = (profit / self.initial_capital * 100)
        
        # Max Drawdown
        peak = self.initial_capital
        max_dd = 0
        for x in self.capital_history:
            if x > peak: peak = x
            dd = (peak - x) / peak
            if dd > max_dd: max_dd = dd

        print("\n--- KẾT QUẢ TOP PLAYER STRATEGY ---")
        print(f"Vốn cuối: ${self.current_capital:.2f} (Lãi: ${profit:.2f})")
        print(f"ROI: {roi:.2f}% | Max Drawdown: {max_dd*100:.2f}%")
        print("-" * 30)
        print(f"Số trận đủ tiêu chuẩn Top {self.top_n}: {total_bets + skip_edge}")
        print(f" -> Đã cược: {total_bets} (Win Rate: {wins/total_bets*100 if total_bets>0 else 0:.1f}%)")
        print(f" -> Bỏ qua (Không thơm): {skip_edge}")
        print(f"Số trận bị loại (Rank thấp): {skip_rank}")
        print("-----------------------------------\n")

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.capital_history, color='darkblue', label='Bankroll')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--')
        plt.title(f'Equity Curve: Top {self.top_n} Only')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

class TwoSidedBacktester:
    def __init__(self, initial_capital=1000, bet_amount=100, threshold=0.05):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.bet_amount = bet_amount
        self.threshold = threshold
        self.capital_history = [initial_capital]
        self.metrics = {}

    def run(self, df, prob_col='Prob_P1', p1_col='Name_1', winner_col='Victory'):
        print(f"--- BẮT ĐẦU BACKTEST 2 CHIỀU (Threshold: {self.threshold*100}%) ---")
        
        bets_won = 0
        bets_lost = 0
        skipped = 0
        
        data = df.copy()
        
        for idx, row in data.iterrows():
            if self.current_capital <= 0:
                print("!!! CHÁY TÀI KHOẢN !!!")
                break
            # 1. LẤY DỮ LIỆU CƠ BẢN
            name_1 = row[p1_col]
            winner = row[winner_col]
            
            # --- PHẦN QUAN TRỌNG: TÍNH TOÁN CẢ 2 ĐẦU ---
            
            # A. Thông tin Player 1
            prob_p1 = row[prob_col]           # Model dự đoán P1 thắng
            
            # B. Thông tin Player 2
            prob_p2 = 1.0 - prob_p1           # Xác suất P2 thắng (Phần bù của P1)


            odds_p1 = row['PS_1']  # Kèo thua dành cho P1
            odds_p2 = row['PS_2']  # Kèo thắng dành cho P2
            if winner == 0:
                # P1 Thắng thực tế
                # odds_p1 = row['PSW']  # Kèo thắng dành cho P1
                # odds_p2 = row['PSL']  # Kèo thua dành cho P2
                p1_outcome = 1        # P1 thắng = 1
            else:
                # P1 Thua thực tế
                # odds_p1 = row['PSL']  # Kèo thua dành cho P1
                # odds_p2 = row['PSW']  # Kèo thắng dành cho P2
                p1_outcome = 0        # P1 thua = 0 (tức là P2 thắng)

            if pd.isna(odds_p1) or pd.isna(odds_p2): continue

            # 2. TÍNH EDGE (LỢI THẾ) CHO CẢ 2 BÊN
            edge_p1 = prob_p1 - (1 / odds_p1)
            edge_p2 = prob_p2 - (1 / odds_p2)
            
            # 3. RA QUYẾT ĐỊNH (CHỌN KÈO THƠM NHẤT)
            # Logic: Chỉ cược vào kèo nào có Edge > Threshold.
            # Nếu cả 2 đều thơm (hiếm), chọn cái thơm hơn.
            
            selected_bet = None # Không cược
            
            if edge_p1 > self.threshold and edge_p1 > edge_p2:
                selected_bet = 'P1'
            elif edge_p2 > self.threshold and edge_p2 > edge_p1:
                selected_bet = 'P2'
            
            # 4. XỬ LÝ KẾT QUẢ CƯỢC
            if selected_bet == 'P1':
                if p1_outcome == 1: # P1 thắng thật
                    profit = self.bet_amount * (odds_p1 - 1)
                    self.current_capital += profit
                    bets_won += 1
                else: # P1 thua
                    self.current_capital -= self.bet_amount
                    bets_lost += 1
                    
            elif selected_bet == 'P2':
                if p1_outcome == 0: # P1 thua = P2 thắng thật
                    profit = self.bet_amount * (odds_p2 - 1)
                    self.current_capital += profit
                    bets_won += 1
                else: # P2 thua
                    self.current_capital -= self.bet_amount
                    bets_lost += 1
            else:
                skipped += 1

            if self.current_capital == self.capital_history[-1]:
                continue
            self.capital_history.append(self.current_capital)
            
        self._generate_report(bets_won, bets_lost, skipped)

    def _generate_report(self, wins, losses, skipped):
        total_bets = wins + losses
        profit = self.current_capital - self.initial_capital
        roi = (profit / (total_bets * self.bet_amount) * 100) if total_bets > 0 else 0
        
        print("\n--- KẾT QUẢ BACKTEST ---")
        print(f"Lợi nhuận: ${profit:.2f}")
        print(f"Tổng lệnh: {total_bets} (Bỏ qua: {skipped})")
        print(f"ROI: {roi:.2f}%")
        print("------------------------")
    
    def plot_equity_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.capital_history, color='green')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--')
        plt.title('Equity Curve (Two-Sided Betting)')
        plt.show()

def trades_df(bt):
    df = pd.DataFrame(getattr(bt, "trades", []))
    if df.empty:
        return df

    # coerce numerics
    for c in ["pnl", "stake", "prob", "odds", "rank1", "rank2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute rank_diff if possible
    if "rank_diff" not in df.columns:
        if "rank1" in df.columns and "rank2" in df.columns:
            df["rank_diff"] = df["rank2"] - df["rank1"]

    # fill labels
    if "tournament_level" in df.columns:
        df["tournament_level"] = df["tournament_level"].fillna("(Unknown)").astype(str)

    return df

def _empty_msg(title, msg):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False)]
    )
    return fig

def fig_profit_by(df_trades, col="tournament_level", title="Profit by Tournament Level"):
    if df_trades is None or df_trades.empty:
        return _empty_msg(title, "No trades logged (bt.trades is empty).")
    if col not in df_trades.columns:
        return _empty_msg(title, f"Missing column: {col}")
    g = df_trades.groupby(col, dropna=False)["pnl"].sum().reset_index()
    g[col] = g[col].fillna("(Unknown)")
    if g.empty:
        return _empty_msg(title, "No data after grouping.")
    fig = px.bar(g.sort_values("pnl", ascending=False), x=col, y="pnl", title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def fig_roi_by_rankdiff(df_trades, title="ROI by Rank Difference Bucket"):
    if df_trades is None or df_trades.empty:
        return _empty_msg(title, "No trades logged.")
    if "rank_diff" not in df_trades.columns:
        return _empty_msg(title, "Missing rank_diff (need rank1/rank2 in trade log).")

    d = df_trades.dropna(subset=["rank_diff"]).copy()
    if d.empty:
        return _empty_msg(title, "rank_diff is all NaN.")

    bins = [-9999, -200, -100, -50, -20, 0, 20, 50, 100, 200, 9999]
    labels = ["<-200","-200:-100","-100:-50","-50:-20","-20:0","0:20","20:50","50:100","100:200",">200"]
    d["rank_bucket"] = pd.cut(d["rank_diff"], bins=bins, labels=labels)

    g = d.groupby("rank_bucket", observed=True).agg(
        pnl=("pnl", "sum"),
        stake=("stake", "sum"),
        bets=("pnl", "size"),
    ).reset_index()
    g["roi"] = (g["pnl"] / g["stake"]).replace([np.inf, -np.inf], np.nan) * 100
    g["roi"] = g["roi"].fillna(0)

    fig = px.bar(g, x="rank_bucket", y="roi", title=title, hover_data=["bets", "pnl", "stake"])
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def fig_pnl_by_prob_bucket(df_trades, title="ROI by Predicted Probability Bucket"):
    def empty_msg(msg):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    if df_trades is None or df_trades.empty:
        return empty_msg("No trades logged.")
    if "prob" not in df_trades.columns:
        return empty_msg("Missing prob in trade log.")

    d = df_trades.copy()
    d["prob"] = pd.to_numeric(d["prob"], errors="coerce")
    d = d.dropna(subset=["prob"])
    if d.empty:
        return empty_msg("prob is all NaN.")

    d["p_bucket"] = pd.cut(d["prob"], bins=np.linspace(0, 1, 11), include_lowest=True)

    g = d.groupby("p_bucket", observed=True).agg(
        pnl=("pnl", "sum"),
        stake=("stake", "sum"),
        bets=("pnl", "size")
    ).reset_index()

    g["roi"] = np.where(g["stake"] > 0, 100.0 * g["pnl"] / g["stake"], 0.0)

    # ✅ IMPORTANT: convert Interval -> string label for Plotly/Dash JSON serialization
    g["p_bucket"] = g["p_bucket"].astype(str)

    fig = px.line(g, x="p_bucket", y="roi", markers=True, title=title)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Predicted probability bucket",
        yaxis_title="ROI (%)",
    )
    return fig