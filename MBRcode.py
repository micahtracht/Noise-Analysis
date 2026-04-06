# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion
class CalculatingFluorescentPinkCaterpillar(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        # Brokerage model for Binance futures
        self.SetBrokerageModel(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)
        # Add spot and future symbols
        self.spot = self.add_crypto("BTCUSDT", Resolution.HOUR).Symbol
        self.future = self.add_crypto_future("BTCUSDT", Resolution.HOUR).Symbol
        # Store latest prices
        self.last_prices = {}
        # Rolling window for basis
        self.window_size = 24
        self.basis_window = RollingWindow[float](self.window_size)
        # Track trade performance
        self.last_trade_time = None
        self.last_portfolio_value = self.portfolio.total_portfolio_value
        self.last_basis_trade_value = None
        # Plot Basis
        self.basis_chart = Chart("Basis")
        self.basis_chart.AddSeries(Series("BasisValue", SeriesType.LINE, 0))
        self.AddChart(self.basis_chart)
        # Cooldown period between trades
        self.cooldown_hours = 6
        # Schedule trading logic hourly
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self.TradeBasis
        )
    def OnData(self, slice):
        if self.spot in slice.Bars:
            self.last_prices[self.spot] = slice.Bars[self.spot].Close
        if self.future in slice.Bars:
            self.last_prices[self.future] = slice.Bars[self.future].Close
    def TradeBasis(self):
        # Ensure price data is available
        if self.spot not in self.last_prices or self.future not in self.last_prices:
            return
        spot_price = self.last_prices[self.spot]
        future_price = self.last_prices[self.future]
        # Compute basis
        basis = float((future_price - spot_price) / spot_price)
        self.basis_window.Add(basis)
        self.Plot("Basis", "BasisValue", basis)
        if not self.basis_window.IsReady:
            return
        # Compute z-score
        arr = list(self.basis_window)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return
        z = (basis - mean) / std
        # Check cooldown
        if self.last_trade_time and (self.Time - self.last_trade_time).total_seconds() < self.cooldown_hours * 3600:
            return
        portfolio_value = self.Portfolio.TotalPortfolioValue
        exposure = portfolio_value * 0.25  # use 25% of portfolio
        future_qty = exposure / future_price
        spot_qty = exposure / spot_price
        # --- TRADING LOGIC ---
        if z > 3 and not self.Portfolio[self.future].IsShort:
            self.Debug("Z>3 → short future / long spot")
            self.Liquidate()
            self.MarketOrder(self.future, -future_qty)
            self.MarketOrder(self.spot, spot_qty)
            self.last_trade_time = self.Time
            self.last_basis_trade_value = self.Portfolio.TotalPortfolioValue
        elif z < -3 and not self.Portfolio[self.future].IsLong:
            self.Debug("Z<-3 → long future / short spot")
            self.Liquidate()
            self.MarketOrder(self.future, future_qty)
            self.MarketOrder(self.spot, -spot_qty)
            self.last_trade_time = self.Time
            self.last_basis_trade_value = self.Portfolio.TotalPortfolioValue
        elif abs(z) < 0.5 and (self.Portfolio[self.future].Invested or self.Portfolio[self.spot].Invested):
            # --- EXIT LOGIC ---
            self.Liquidate()
            if self.last_basis_trade_value:
                trade_profit = self.Portfolio.TotalPortfolioValue - self.last_basis_trade_value
                self.Debug(f"Z≈0 → Closed positions | Profit: {trade_profit:.2f} USDT")
            self.last_trade_time = self.Time
            self.last_basis_trade_value = None
