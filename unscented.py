from AlgorithmImports import *
import numpy as np

class UnscentedKalmanCaterpillar(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour, Market.Binance).Symbol

        # --- UNSCENTED KALMAN FILTER PARAMETERS ---
        # State vector: [basis level, basis velocity]
        self.state = np.zeros(2)
        self.covar = np.eye(2)
        
        self.Q = np.array([
            [1e-5, 0.0],
            [0.0, 1e-6]
        ])
        self.R = 1e-3
        
        self.state_dim = 2
        self.alpha = 0.1
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = (self.alpha ** 2) * (self.state_dim + self.kappa) - self.state_dim
        
        self.mean_weights = np.ones(2 * self.state_dim + 1) / (2 * (self.state_dim + self.lambda_))
        self.covar_weights = self.mean_weights.copy()
        self.mean_weights[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.covar_weights[0] = self.mean_weights[0] + (1 - self.alpha ** 2 + self.beta)
        
        self.basis_history = RollingWindow[float](24)
        
        self.last_prices = {}
        self.last_trade_time = datetime.min
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.TradeBasis)

    def StateTransition(self, state):
        """Second order direct model: next basis is level plus velocity."""
        level = state[0]
        velocity = state[1]
        return np.array([level + velocity, velocity])

    def ObservationModel(self, state):
        """The observed basis is the state level."""
        return state[0]

    def GetSigmaPoints(self, state, covar):
        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = state

        scaled_covar = (self.state_dim + self.lambda_) * covar
        
        try:
            sqrt_covar = np.linalg.cholesky(scaled_covar)
        except:
            sqrt_covar = np.linalg.cholesky(scaled_covar + np.eye(self.state_dim) * 1e-12)

        for i in range(self.state_dim):
            sigma_points[i + 1] = state + sqrt_covar[:, i]
            sigma_points[i + 1 + self.state_dim] = state - sqrt_covar[:, i]

        return sigma_points

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
        sigma_points = self.GetSigmaPoints(self.state, self.covar)
        pred_sigma_points = np.array([self.StateTransition(point) for point in sigma_points])
        
        state_pred = np.sum(self.mean_weights[:, None] * pred_sigma_points, axis=0)
        covar_pred = self.Q.copy()
        
        for i in range(2 * self.state_dim + 1):
            diff = pred_sigma_points[i] - state_pred
            covar_pred += self.covar_weights[i] * np.outer(diff, diff)

        basis_prediction = state_pred[0]

        # --- 2. EVALUATION & PLOTTING ---
        std = np.std(hist_vec)
        z = (current_basis - basis_prediction) / std if std > 0 else 0

        self.Plot("Unscented KF Analysis", "Basis Prediction", basis_prediction)
        self.Plot("Unscented KF Analysis", "Actual Basis", current_basis)
        self.Plot("Unscented KF Analysis", "Velocity", state_pred[1])
        self.Plot("Z-Score", "Value", z)

        # --- 3. UNSCENTED KALMAN UPDATE ---
        obs_sigma_points = np.array([self.ObservationModel(point) for point in pred_sigma_points])
        obs_pred = np.sum(self.mean_weights * obs_sigma_points)
        
        innovation_covar = dynamic_R
        cross_covar = np.zeros(self.state_dim)
        
        for i in range(2 * self.state_dim + 1):
            state_diff = pred_sigma_points[i] - state_pred
            obs_diff = obs_sigma_points[i] - obs_pred
            innovation_covar += self.covar_weights[i] * obs_diff * obs_diff
            cross_covar += self.covar_weights[i] * state_diff * obs_diff

        innovation = current_basis - obs_pred
        gain = cross_covar / innovation_covar
        
        self.state = state_pred + gain * innovation
        self.covar = covar_pred - np.outer(gain, gain) * innovation_covar
        self.covar = 0.5 * (self.covar + self.covar.T)

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
