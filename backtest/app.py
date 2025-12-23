import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from backtest_utils import TwoSidedKellyBacktester, TopPlayerKellyBacktester, TwoSidedBacktester, fig_calibration, fig_pnl_by_edge, fig_roi_by_odds, fig_prob_hist

# Khởi tạo app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Tạo dữ liệu mẫu
def read_output_model():
    test_df = pd.read_csv('test_df.csv')
    return test_df

# Layout chính với Tabs
app.layout = html.Div([
    html.H1('🎾 Tennis Betting Backtest Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30, 
                   'padding': '20px', 'backgroundColor': '#ecf0f1'}),
    
    dcc.Tabs(id='main-tabs', value='backtest-tab', children=[
        dcc.Tab(label='📊 Backtest Strategies', value='backtest-tab', 
                style={'fontWeight': 'bold', 'fontSize': 16},
                selected_style={'fontWeight': 'bold', 'fontSize': 16, 'color': '#27ae60'}),
        dcc.Tab(label='📈 Dashboard Overview', value='dashboard-tab',
                style={'fontWeight': 'bold', 'fontSize': 16},
                selected_style={'fontWeight': 'bold', 'fontSize': 16, 'color': '#3498db'}),
    ], style={'marginBottom': 30}),
    
    html.Div(id='tab-content')
    
], style={
    'maxWidth': '1400px',
    'margin': '0 auto',
    'padding': '20px',
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#f5f6fa'
})

# Callback để render tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'backtest-tab':
        return create_backtest_tab()
    elif tab == 'dashboard-tab':
        return create_dashboard_tab()

def create_backtest_tab():
    return html.Div([
        # Store để lưu parameters
        dcc.Store(id='params-store'),
        
        html.Div([
            # Strategy Selector
            html.Div([
                html.Label('Chọn Chiến Thuật:', style={'fontWeight': 'bold', 'fontSize': 16}),
                dcc.Dropdown(
                    id='strategy-dropdown',
                    options=[
                        {'label': '📊 Two-Sided Kelly', 'value': 'kelly'},
                        {'label': '🏆 Top Player Kelly', 'value': 'top_player'},
                        {'label': '💰 Two-Sided Simple', 'value': 'simple'}
                    ],
                    value='kelly',
                    style={'width': '100%'}
                )
            ], style={'marginBottom': 30}),
            
            # Parameters Panel
            html.Div(id='params-panel', style={'marginBottom': 30}),
            
            # Run Button
            html.Div([
                html.Button('🚀 Running Backtest', id='run-button', n_clicks=0,
                           style={
                               'width': '100%',
                               'padding': '15px',
                               'fontSize': 18,
                               'fontWeight': 'bold',
                               'backgroundColor': '#27ae60',
                               'color': 'white',
                               'border': 'none',
                               'borderRadius': '8px',
                               'cursor': 'pointer'
                           })
            ], style={'marginBottom': 30}),
            
            # Loading Spinner
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    html.Div(id='results-cards', style={'marginBottom': 30}),
                    
                    # Equity Chart
                    dcc.Graph(id='equity-chart'),
                    
                    # Stats Table
                    html.Div(id='stats-table')
                ]
            )
        ])
    ])

# TAB 2: DASHBOARD OVERVIEW (Trống - để custom sau)
def create_dashboard_tab():
    return html.Div([
        html.H2('📈 Dashboard Overview', 
               style={'color': '#2c3e50', 'marginBottom': 30, 'textAlign': 'center'}),
        html.P('Tab này để trống - bạn có thể thêm nội dung tùy chỉnh ở đây',
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18})
    ])

# Callback để cập nhật parameters panel (TAB 1)
@app.callback(
    Output('params-panel', 'children'),
    Input('strategy-dropdown', 'value')
)
def update_params_panel(strategy):
    common_style = {'marginBottom': 15}
    label_style = {'fontWeight': 'bold', 'marginBottom': 5}
    
    if strategy == 'kelly':
        return html.Div([
            html.H3('⚙️ Two-Sided Kelly Hyperparameters', style={'color': '#8e44ad'}),
            html.Div([
                html.Label('Initial Capital ($):', style=label_style),
                dcc.Input(id={'type': 'param', 'name': 'capital'}, type='number', value=1000, 
                         style={'width': '100%', 'padding': '8px'})
            ], style=common_style),
            html.Div([
                html.Label('Kelly Multiplier (0-1):', style=label_style),
                dcc.Slider(id={'type': 'param', 'name': 'kelly-mult'}, 
                          min=0.1, max=1.0, step=0.1, value=0.5, 
                          marks={i/10: str(i/10) for i in range(1, 11)})
            ], style=common_style),
            html.Div([
                html.Label('Max Stake % (0-0.3):', style=label_style),
                dcc.Slider(id={'type': 'param', 'name': 'max-stake'}, 
                          min=0.05, max=0.3, step=0.05, value=0.15, 
                          marks={i/100: f'{i}%' for i in range(5, 35, 5)})
            ], style=common_style),
        ], style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 8})
    
    elif strategy == 'top_player':
        return html.Div([
            html.H3('⚙️ Top Player Strategy Hyperparameters', style={'color': '#3498db'}),
            html.Div([
                html.Label('Initial Capital ($):', style=label_style),
                dcc.Input(id={'type': 'param', 'name': 'capital'}, type='number', value=1000, 
                         style={'width': '100%', 'padding': '8px'})
            ], style=common_style),
            html.Div([
                html.Label('Top N Players:', style=label_style),
                dcc.Slider(id={'type': 'param', 'name': 'top-n'}, 
                          min=5, max=20, step=1, value=8,
                          marks={i: str(i) for i in range(5, 21, 5)})
            ], style=common_style),
            html.Div([
                html.Label('Kelly Multiplier (0-1):', style=label_style),
                dcc.Slider(id={'type': 'param', 'name': 'kelly-mult'}, 
                          min=0.1, max=1.0, step=0.1, value=0.5, 
                          marks={i/10: str(i/10) for i in range(1, 11)})
            ], style=common_style),
            html.Div([
                html.Label('Max Stake % (0-0.3):', style=label_style),
                dcc.Slider(id={'type': 'param', 'name': 'max-stake'}, 
                          min=0.05, max=0.3, step=0.05, value=0.15, 
                          marks={i/100: f'{i}%' for i in range(5, 35, 5)})
            ], style=common_style),
        ], style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 8})
    
    else:  # simple
        return html.Div([
            html.H3('⚙️ Simple Strategy Hyperparameters', style={'color': '#e74c3c'}),
            html.Div([
                html.Label('Initial Capital ($):', style=label_style),
                dcc.Input(id={'type': 'param', 'name': 'capital'}, type='number', value=1000, 
                         style={'width': '100%', 'padding': '8px'})
            ], style=common_style),
            html.Div([
                html.Label('Bet Amount ($):', style=label_style),
                dcc.Input(id={'type': 'param', 'name': 'bet-amount'}, type='number', value=100, 
                         style={'width': '100%', 'padding': '8px'})
            ], style=common_style),
            html.Div([
                html.Label('Edge Threshold (0-0.2):', style=label_style),
                dcc.Slider(id={'type': 'param', 'name': 'threshold'}, 
                          min=0.01, max=0.2, step=0.01, value=0.05, 
                          marks={i/100: f'{i}%' for i in range(0, 25, 5)})
            ], style=common_style),
        ], style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 8})

# Callback để lưu parameters vào store (TAB 1)
@app.callback(
    Output('params-store', 'data'),
    [Input({'type': 'param', 'name': dash.dependencies.ALL}, 'value')],
    [State('strategy-dropdown', 'value')]
)
def save_params(param_values, strategy):
    ctx = callback_context
    if not ctx.triggered:
        return {}
    
    param_names = [item['id']['name'] for item in ctx.inputs_list[0]]
    params = dict(zip(param_names, param_values))
    params['strategy'] = strategy
    
    return params

# Callback chính để chạy backtest (TAB 1)
@app.callback(
    [Output('results-cards', 'children'),
     Output('equity-chart', 'figure'),
     Output('stats-table', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('params-store', 'data'),
     State('strategy-dropdown', 'value')]
)
def run_backtest(n_clicks, params, strategy):
    
    if n_clicks == 0 or not params:
        return html.Div(), {}, html.Div()
    
    df = read_output_model()
    capital = params.get('capital', 1000)
    
    if strategy == 'kelly':
        bt = TwoSidedKellyBacktester(
            initial_capital=capital,
            kelly_multiplier=params.get('kelly-mult', 0.5),
            max_stake_pct=params.get('max-stake', 0.15)
        )
        bt.run(df)
        trades = bt.trades_df  # created as above

        fig_cal = fig_calibration(trades)
        fig_edge = fig_pnl_by_edge(trades)
        fig_odds = fig_roi_by_odds(trades)
        fig_hist = fig_prob_hist(trades)
        title = 'Two-Sided Kelly Strategy'
        color = '#8e44ad'
        
    elif strategy == 'top_player':
        bt = TopPlayerKellyBacktester(
            top_n=int(params.get('top-n', 8)),
            initial_capital=capital,
            kelly_multiplier=params.get('kelly-mult', 0.5),
            max_stake_pct=params.get('max-stake', 0.15)
        )
        bt.run(df)
        title = f'Top {int(params.get("top-n", 8))} Player Strategy'
        color = '#3498db'
        
    else:
        bt = TwoSidedBacktester(
            initial_capital=capital,
            bet_amount=params.get('bet-amount', 100),
            threshold=params.get('threshold', 0.05)
        )
        bt.run(df)
        title = 'Two-Sided Simple Strategy'
        color = '#e74c3c'
    
    profit = bt.current_capital - bt.initial_capital
    roi = (profit / bt.initial_capital) * 100
    
    peak = bt.initial_capital
    max_dd = 0
    for x in bt.capital_history:
        if x > peak: peak = x
        dd = (peak - x) / peak
        if dd > max_dd: max_dd = dd
    
    cards = html.Div([
        html.Div([
            html.H4('💵 Profit', style={'color': '#7f8c8d', 'marginBottom': 10}),
            html.H2(f'${profit:,.2f}', 
                   style={'color': '#27ae60' if profit > 0 else '#e74c3c', 'margin': 0})
        ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#ecf0f1', 
                 'borderRadius': 8, 'textAlign': 'center', 'marginRight': 10}),
        
        html.Div([
            html.H4('📈 ROI', style={'color': '#7f8c8d', 'marginBottom': 10}),
            html.H2(f'{roi:.2f}%', 
                   style={'color': '#27ae60' if roi > 0 else '#e74c3c', 'margin': 0})
        ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#ecf0f1', 
                 'borderRadius': 8, 'textAlign': 'center', 'marginRight': 10}),
        
        html.Div([
            html.H4('💰 Final capital', style={'color': '#7f8c8d', 'marginBottom': 10}),
            html.H2(f'${bt.current_capital:,.2f}', 
                   style={'color': '#2c3e50', 'margin': 0})
        ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#ecf0f1', 
                 'borderRadius': 8, 'textAlign': 'center', 'marginRight': 10}),
        
        html.Div([
            html.H4('📉 Max DD', style={'color': '#7f8c8d', 'marginBottom': 10}),
            html.H2(f'{max_dd*100:.2f}%', 
                   style={'color': '#e74c3c', 'margin': 0})
        ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#ecf0f1', 
                 'borderRadius': 8, 'textAlign': 'center'})
        
    ], style={'display': 'flex', 'marginBottom': 30})
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=bt.capital_history,
        mode='lines',
        name='Equity',
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
    ))
    
    fig.add_hline(y=bt.initial_capital, line_dash="dash", 
                  line_color="red", annotation_text="Initial Capital")
    
    fig.update_layout(
        title=dict(text=f'📊 Equity Curve - {title}', x=0.5, xanchor='center'),
        xaxis_title='Number of Matches',
        yaxis_title='Capital ($)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    stats = html.Div([
        html.H3('📋 Detail Statistics', style={'marginBottom': 20}),
        html.Table([
            html.Tr([
                html.Th('Information', style={'padding': 10, 'textAlign': 'left', 'backgroundColor': '#34495e', 'color': 'white'}), 
                html.Th('Value', style={'padding': 10, 'textAlign': 'right', 'backgroundColor': '#34495e', 'color': 'white'})
            ]),
            html.Tr([html.Td('Initial Capital', style={'padding': 10}), 
                    html.Td(f'${bt.initial_capital:,.2f}', style={'padding': 10, 'textAlign': 'right'})]),
            html.Tr([html.Td('Final Capital', style={'padding': 10, 'backgroundColor': '#ecf0f1'}), 
                    html.Td(f'${bt.current_capital:,.2f}', style={'padding': 10, 'textAlign': 'right', 'backgroundColor': '#ecf0f1'})]),
            html.Tr([html.Td('Profit', style={'padding': 10}), 
                    html.Td(f'${profit:,.2f}', style={'padding': 10, 'textAlign': 'right', 
                    'color': '#27ae60' if profit > 0 else '#e74c3c', 'fontWeight': 'bold'})]),
            html.Tr([html.Td('ROI', style={'padding': 10, 'backgroundColor': '#ecf0f1'}), 
                    html.Td(f'{roi:.2f}%', style={'padding': 10, 'textAlign': 'right', 'backgroundColor': '#ecf0f1',
                    'color': '#27ae60' if roi > 0 else '#e74c3c', 'fontWeight': 'bold'})]),
            html.Tr([html.Td('Max Drawdown', style={'padding': 10}), 
                    html.Td(f'{max_dd*100:.2f}%', style={'padding': 10, 'textAlign': 'right', 
                    'color': '#e74c3c', 'fontWeight': 'bold'})]),
            html.Tr([html.Td('Peak Capital', style={'padding': 10, 'backgroundColor': '#ecf0f1'}), 
                    html.Td(f'${peak:,.2f}', style={'padding': 10, 'textAlign': 'right', 'backgroundColor': '#ecf0f1'})]),
            html.Tr([html.Td('Number of Bettings Matches', style={'padding': 10}), 
                    html.Td(str(len(df)), style={'padding': 10, 'textAlign': 'right'})]),
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'backgroundColor': 'white',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ], style={
        'padding': 20,
        'backgroundColor': '#f8f9fa',
        'borderRadius': 8
    })
    
    return cards, fig, stats

# Callback để chạy comparison (TAB 2)
@app.callback(
    [Output('comparison-cards', 'children'),
     Output('comparison-equity-chart', 'figure'),
     Output('comparison-roi-chart', 'figure'),
     Output('comparison-dd-chart', 'figure')],
    [Input('compare-button', 'n_clicks')],
    [State('dash-n-matches', 'value'),
     State('dash-capital', 'value')]
)
def run_comparison(n_clicks, n_matches, capital):
    if n_clicks == 0:
        return html.Div(), {}, {}, {}
    
    df = read_output_model()
    capital = capital or 1000
    
    # Chạy cả 3 strategies
    results = {}
    
    # Kelly
    bt_kelly = TwoSidedKellyBacktester(initial_capital=capital, kelly_multiplier=0.5, max_stake_pct=0.15)
    bt_kelly.run(df)
    results['Kelly'] = {
        'bt': bt_kelly,
        'color': '#8e44ad',
        'profit': bt_kelly.current_capital - capital,
        'roi': (bt_kelly.current_capital - capital) / capital * 100
    }
    
    # Top Player
    bt_top = TopPlayerKellyBacktester(top_n=8, initial_capital=capital, kelly_multiplier=0.5, max_stake_pct=0.15)
    bt_top.run(df)
    results['Top Player'] = {
        'bt': bt_top,
        'color': '#3498db',
        'profit': bt_top.current_capital - capital,
        'roi': (bt_top.current_capital - capital) / capital * 100
    }
    
    # Simple
    bt_simple = TwoSidedBacktester(initial_capital=capital, bet_amount=100, threshold=0.05)
    bt_simple.run(df)
    results['Simple'] = {
        'bt': bt_simple,
        'color': '#e74c3c',
        'profit': bt_simple.current_capital - capital,
        'roi': (bt_simple.current_capital - capital) / capital * 100
    }
    
    # Calculate max drawdown for each
    for name, data in results.items():
        peak = capital
        max_dd = 0
        for x in data['bt'].capital_history:
            if x > peak: peak = x
            dd = (peak - x) / peak
            if dd > max_dd: max_dd = dd
        data['max_dd'] = max_dd * 100
    
    # Comparison Cards
    cards = []
    for name, data in results.items():
        card = html.Div([
            html.H3(name, style={'color': data['color'], 'marginBottom': 15}),
            html.Div([
                html.Div([
                    html.P('Profit', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': 12}),
                    html.H4(f"${data['profit']:,.2f}", 
                           style={'margin': '5px 0', 'color': '#27ae60' if data['profit'] > 0 else '#e74c3c'})
                ], style={'marginBottom': 10}),
                html.Div([
                    html.P('ROI', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': 12}),
                    html.H4(f"{data['roi']:.2f}%", 
                           style={'margin': '5px 0', 'color': '#27ae60' if data['roi'] > 0 else '#e74c3c'})
                ], style={'marginBottom': 10}),
                html.Div([
                    html.P('Max DD', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': 12}),
                    html.H4(f"{data['max_dd']:.2f}%", 
                           style={'margin': '5px 0', 'color': '#e74c3c'})
                ])
            ])
        ], style={
            'flex': 1,
            'padding': 25,
            'backgroundColor': 'white',
            'borderRadius': 10,
            'marginRight': 15,
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'border': f'3px solid {data["color"]}'
        })
        cards.append(card)
    
    cards_div = html.Div(cards, style={'display': 'flex', 'marginBottom': 30})
    
    # Equity Comparison Chart
    equity_fig = go.Figure()
    for name, data in results.items():
        equity_fig.add_trace(go.Scatter(
            y=data['bt'].capital_history,
            mode='lines',
            name=name,
            line=dict(color=data['color'], width=3)
        ))
    
    equity_fig.add_hline(y=capital, line_dash="dash", line_color="red", annotation_text="Vốn Gốc")
    equity_fig.update_layout(
        title='📊 So Sánh Equity Curves',
        xaxis_title='Số Trận',
        yaxis_title='Vốn ($)',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    # ROI Bar Chart
    roi_fig = go.Figure(data=[
        go.Bar(
            x=list(results.keys()),
            y=[data['roi'] for data in results.values()],
            marker_color=[data['color'] for data in results.values()],
            text=[f"{data['roi']:.2f}%" for data in results.values()],
            textposition='outside'
        )
    ])
    roi_fig.update_layout(
        title='📈 So Sánh ROI',
        yaxis_title='ROI (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    # Max DD Bar Chart
    dd_fig = go.Figure(data=[
        go.Bar(
            x=list(results.keys()),
            y=[data['max_dd'] for data in results.values()],
            marker_color=[data['color'] for data in results.values()],
            text=[f"{data['max_dd']:.2f}%" for data in results.values()],
            textposition='outside'
        )
    ])
    dd_fig.update_layout(
        title='📉 So Sánh Max Drawdown',
        yaxis_title='Max DD (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return cards_div, equity_fig, roi_fig, dd_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)