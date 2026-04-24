from AlgorithmImports import *
import numpy as np

class FractionalKalmanCaterpillar(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour, Market.Binance).Symbol

        # --- FRACTIONAL PARAMETERS ---
        # 10 Candidate Branches evenly spaced between 0.1 and 0.9
        self.deltas = np.linspace(0.1, 0.9, 10).tolist()
        self.num_models = len(self.deltas)
        
        # Generates arrays of varying lengths based on mathematical precision
        self.coeffs = [self.GetDynamicGLCoefficients(d, tolerance=1e-4, max_L=720) for d in self.deltas]
        
        # The rolling window must hold enough data for the longest-memory branch
        self.max_L = max([len(c) for c in self.coeffs])
        self.Debug(f"Maximum required memory window: {self.max_L} hours.")
        
        self.states = np.zeros(self.num_models)
        self.covars = np.ones(self.num_models)
        self.weights = np.ones(self.num_models) / self.num_models
        
        self.basis_history = RollingWindow[float](self.max_L)
        
        self.Q = 1e-5 
        self.R = 1e-3 
        
        self.last_prices = {}
        self.last_trade_time = datetime.min
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.TradeBasis)

    def GetDynamicGLCoefficients(self, delta, tolerance, max_L):
        """Generates autoregressive weights until the decay drops below the tolerance."""
        w = [1.0] # Expansion starts at 1.0
        k = 1
        
        while True:
            next_w = w[-1] * (1 - (delta + 1) / k)
            w.append(next_w)
            
            # Stop if the weight is statistically insignificant or we hit the safety limit
            if abs(next_w) < tolerance or k >= max_L:
                break
            k += 1
            
        # Return -w[1:] to predict X_t using X_{t-1}, X_{t-2}...
        return -np.array(w[1:])

    def OnData(self, slice):
        if self.spot in slice.Bars: self.last_prices[self.spot] = slice.Bars[self.spot].Close
        if self.future in slice.Bars: self.last_prices[self.future] = slice.Bars[self.future].Close

    def TradeBasis(self):
        if self.spot not in self.last_prices or self.future not in self.last_prices: return

        current_basis = float((self.last_prices[self.future] - self.last_prices[self.spot]) / self.last_prices[self.spot])
        
        if not self.basis_history.IsReady: 
            self.basis_history.Add(current_basis)
            return

        hist_vec = np.array([x for x in self.basis_history])
        dynamic_R = np.var(hist_vec) if np.var(hist_vec) > 0 else 1e-5

        # --- 1. A PRIORI PREDICTION (Strictly No Bias) ---
        predictions = np.zeros(self.num_models)
        for i in range(self.num_models):
            # Dynamically slice the history to match the specific memory length of this branch
            L_i = len(self.coeffs[i])
            branch_hist = hist_vec[:L_i]
            predictions[i] = np.dot(self.coeffs[i], branch_hist)

        fused_prediction = np.dot(self.weights, predictions)

        # --- 2. EVALUATION & PLOTTING ---
        std = np.std(hist_vec)
        z = (current_basis - fused_prediction) / std if std > 0 else 0

        winning_delta = self.deltas[np.argmax(self.weights)]
        self.Plot("Fractional Analysis", "Fused Prediction", fused_prediction)
        self.Plot("Fractional Analysis", "Actual Basis", current_basis)
        self.Plot("Leaderboard", "Best Delta", winning_delta)
        self.Plot("Z-Score", "Value", z)

        # --- 3. KALMAN UPDATE & BAYESIAN SCORING ---
        likelihoods = np.zeros(self.num_models)
        for i in range(self.num_models):
            p_pred = self.covars[i] + self.Q
            
            innovation = current_basis - predictions[i]
            gain = p_pred / (p_pred + dynamic_R)
            
            self.states[i] = predictions[i] + gain * innovation
            self.covars[i] = (1 - gain) * p_pred

            likelihoods[i] = (1 / np.sqrt(2 * np.pi * (p_pred + dynamic_R))) * \
                             np.exp(-0.5 * (innovation**2) / (p_pred + dynamic_R))

        self.weights *= likelihoods
        self.weights += 1e-4 
        self.weights /= np.sum(self.weights) 

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
