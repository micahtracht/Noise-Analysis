from AlgorithmImports import *
import numpy as np

class DirectSecondOrderKalmanCaterpillar(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour, Market.Binance).Symbol

        # --- SECOND ORDER EKF PARAMETERS ---
        # State vector: [basis level, basis velocity]
        self.state = np.zeros(2)
        self.covar = np.eye(2)
        
        self.Q = np.array([
            [1e-5, 0.0],
            [0.0, 1e-6]
        ])
        self.R = 1e-3
        
        self.basis_history = RollingWindow[float](24)
        
        self.last_prices = {}
        self.last_trade_time = datetime.min
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.TradeBasis)

    def StateTransition(self, state):
        """Second order direct model: next basis is level plus velocity."""
        level = state[0]
        velocity = state[1]
        return np.array([level + velocity, velocity])

    def StateJacobian(self, state):
        """Jacobian of the second order state transition."""
        return np.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])

    def ObservationModel(self, state):
        """The observed basis is the state level."""
        return state[0]

    def ObservationJacobian(self, state):
        """Jacobian of the basis observation model."""
        return np.array([[1.0, 0.0]])

    def OnData(self, slice):
        if self.spot in slice.Bars: self.last_prices[self.spot] = slice.Bars[self.spot].Close
        if self.future in slice.Bars: self.last_prices[self.future] = slice.Bars[self.future].Close

    def TradeBasis(self):
        if self.spot not in self.last_prices or self.future not in self.last_prices: return

        current_basis = float((self.last_prices[self.future] - self.last_prices[self.spot]) / self.last_prices[self.spot])
        
        if not self.basis_history.IsReady: 
            self.basis_history.Add(current_basis)
            self.state[0] = current_basis
            return

        hist_vec = np.array([x for x in self.basis_history])
        dynamic_R = np.var(hist_vec) if np.var(hist_vec) > 0 else 1e-5

        # --- 1. A PRIORI PREDICTION (Strictly No Bias) ---
        F = self.StateJacobian(self.state)
        state_pred = self.StateTransition(self.state)
        covar_pred = F @ self.covar @ F.T + self.Q

        basis_prediction = state_pred[0]

        # --- 2. EVALUATION & PLOTTING ---
        std = np.std(hist_vec)
        z = (current_basis - basis_prediction) / std if std > 0 else 0

        self.Plot("Second Order Analysis", "Basis Prediction", basis_prediction)
        self.Plot("Second Order Analysis", "Actual Basis", current_basis)
        self.Plot("Second Order Analysis", "Velocity", state_pred[1])
        self.Plot("Z-Score", "Value", z)

        # --- 3. EKF UPDATE ---
        H = self.ObservationJacobian(state_pred)
        innovation = current_basis - self.ObservationModel(state_pred)
        innovation_covar = H @ covar_pred @ H.T + dynamic_R
        gain = covar_pred @ H.T @ np.linalg.inv(innovation_covar)
        
        self.state = state_pred + (gain.flatten() * innovation)
        self.covar = (np.eye(2) - gain @ H) @ covar_pred

        # --- 4. ADVANCE TIME ---
        self.basis_history.Add(current_basis)

        # --- EXECUTION ---
        if abs(z) < 0.5 and self.Portfolio.Invested:
            self.Liquidate()
            # Reset timer on liquidation so we can re-enter immediately if a new signal fires
            self.last_trade_time = datetime.min 
            return
        if (self.Time - self.last_trade_time).total_seconds() < 6 * 3600: return

        if z > 3 and not self.Portfolio.Invested:
            # Entering the market with a market order as a Taker. 
            # Note: We are crossing the spread here, paying the taker fee and the spread cost.
            self.SetHoldings(self.future, -0.2)
            self.SetHoldings(self.spot, 0.2)
            self.last_trade_time = self.Time
        elif z < -3 and not self.Portfolio.Invested:
            self.SetHoldings(self.future, 0.2)
            self.SetHoldings(self.spot, -0.2)
            self.last_trade_time = self.Time
        elif abs(z) < 0.5 and self.Portfolio.Invested:
            self.Liquidate()
            self.last_trade_time = self.Time
