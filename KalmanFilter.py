# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class KalmanBasisCaterpillar(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour, Market.Binance).Symbol

        # --- KALMAN FILTER PARAMETERS ---
        self.state_estimate = 0.0  # Initial guess of the basis mean
        self.error_covariance = 1.0 
        self.process_variance = 1e-5 # Q: How much we allow the "true" mean to drift
        self.measurement_variance = 1e-3 # R: How much noise is in the price data
        
        # We still need a window to calculate a rolling Standard Deviation for the Z-score
        self.basis_window = RollingWindow[float](24)
        
        self.last_prices = {}
        self.cooldown_hours = 6
        self.last_trade_time = datetime.min

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.TradeBasis)

    def OnData(self, slice):
        if self.spot in slice.Bars: self.last_prices[self.spot] = slice.Bars[self.spot].Close
        if self.future in slice.Bars: self.last_prices[self.future] = slice.Bars[self.future].Close

    def TradeBasis(self):
        if self.spot not in self.last_prices or self.future not in self.last_prices:
            return

        spot_p = self.last_prices[self.spot]
        fut_p = self.last_prices[self.future]
        current_basis = float((fut_p - spot_p) / spot_p)
        
        # --- KALMAN UPDATE STEP ---
        # 1. Predict (Time Update)
        # In a stationary model, we assume mean doesn't change, but uncertainty grows slightly
        self.error_covariance += self.process_variance

        # 2. Update (Measurement Update)
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.state_estimate += kalman_gain * (current_basis - self.state_estimate)
        self.error_covariance *= (1 - kalman_gain)

        # Use the Kalman estimate as the "Mean"
        kalman_mean = self.state_estimate
        
        # Store basis to calculate volatility (Std Dev)
        self.basis_window.Add(current_basis)
        if not self.basis_window.IsReady: return

        # Calculate Z-Score using Kalman Mean vs Rolling Std Dev
        std = np.std([x for x in self.basis_window])
        z = (current_basis - kalman_mean) / std if std > 0 else 0

        # Plotting
        self.Plot("Basis Analysis", "Basis", current_basis)
        self.Plot("Basis Analysis", "Kalman Mean", kalman_mean)
        self.Plot("Z-Score", "Value", z)

        # --- EXECUTION LOGIC ---
        if (self.Time - self.last_trade_time).total_seconds() < self.cooldown_hours * 3600:
            return

        if z > 3 and not self.Portfolio.Invested:
            self.SetHoldings(self.future, -0.2); self.SetHoldings(self.spot, 0.2)
            self.last_trade_time = self.Time
        elif z < -3 and not self.Portfolio.Invested:
            self.SetHoldings(self.future, 0.2); self.SetHoldings(self.spot, -0.2)
            self.last_trade_time = self.Time
        elif abs(z) < 0.5 and self.Portfolio.Invested:
            self.Liquidate()
            self.last_trade_time = self.Time
