from AlgorithmImports import *
import numpy as np

class FractionalKalmanCaterpillar(QCAlgorithm):
    def Initialize(self):
        # Recommend keeping this under 4-5 months if you want the charts to fully render
        self.SetStartDate(2024, 9, 1)
        self.SetEndDate(2024, 12, 31) 
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        self.spot = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance).Symbol
        self.future = self.AddCryptoFuture("BTCUSDT", Resolution.Hour, Market.Binance).Symbol

        # --- 1. FRACTIONAL PARAMETERS ---
        self.deltas = np.linspace(0.1, 0.9, 10).tolist()
        self.num_models = len(self.deltas)
        self.coeffs = [self.GetDynamicGLCoefficients(d, tolerance=1e-4, max_L=720) for d in self.deltas]
        self.max_L = max([len(c) for c in self.coeffs])
        
        self.states = np.zeros(self.num_models)
        self.covars = np.ones(self.num_models)
        self.weights = np.ones(self.num_models) / self.num_models
        
        # --- 2. STANDARD KALMAN PARAMETERS (1D) ---
        self.std_kf_state = 0.0
        self.std_kf_covar = 1.0

        # --- 3. EWMA PARAMETERS ---
        self.ewma_state = 0.0
        self.ewma_alpha = 0.05

        # --- 4. SMA PARAMETERS ---
        self.sma_period = 24

        # --- 5. 2D PARTICLE FILTER PARAMETERS ---
        self.num_particles = 100
        # 2D Array: Column 0 = Level (Position), Column 1 = Velocity
        self.particles = np.zeros((self.num_particles, 2)) 
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
        self.pf_initialized = False

        # --- ERROR TRACKING (SSE) ---
        self.fractional_sse = 0.0  
        self.standard_sse = 0.0
        self.ewma_sse = 0.0
        self.sma_sse = 0.0
        self.pf_sse = 0.0
        self.prediction_count = 0

        self.basis_history = RollingWindow[float](self.max_L)
        
        self.Q = 1e-5 
        self.R = 1e-3 
        
        self.last_prices = {}
        self.last_trade_time = datetime.min
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=1)), self.TradeBasis)

    def GetDynamicGLCoefficients(self, delta, tolerance, max_L):
        w = [1.0] 
        k = 1
        while True:
            next_w = w[-1] * (1 - (delta + 1) / k)
            w.append(next_w)
            if abs(next_w) < tolerance or k >= max_L:
                break
            k += 1
        return -np.array(w[1:])

    def OnData(self, slice):
        if self.spot in slice.Bars: self.last_prices[self.spot] = slice.Bars[self.spot].Close
        if self.future in slice.Bars: self.last_prices[self.future] = slice.Bars[self.future].Close

    def TradeBasis(self):
        if self.spot not in self.last_prices or self.future not in self.last_prices: return

        current_basis = float((self.last_prices[self.future] - self.last_prices[self.spot]) / self.last_prices[self.spot])
        
        # WARMUP PHASE
        if not self.basis_history.IsReady: 
            self.basis_history.Add(current_basis)
            self.std_kf_state = current_basis
            self.ewma_state = current_basis
            if not self.pf_initialized:
                # Initialize particles with current basis and 0 velocity
                self.particles[:, 0] = current_basis
                self.particles[:, 1] = 0.0
                self.pf_initialized = True
            return

        hist_vec = np.array([x for x in self.basis_history])
        dynamic_R = np.var(hist_vec) if np.var(hist_vec) > 0 else 1e-5

        # ==========================================
        # 1. A PRIORI PREDICTIONS (STRICTLY NO BIAS)
        # ==========================================
        
        # 1a. Fractional Prediction
        predictions = np.zeros(self.num_models)
        for i in range(self.num_models):
            L_i = len(self.coeffs[i])
            branch_hist = hist_vec[:L_i]
            predictions[i] = np.dot(self.coeffs[i], branch_hist)
        frac_pred = np.dot(self.weights, predictions)

        # 1b. Standard KF Prediction
        std_pred = self.std_kf_state 

        # 1c. EWMA Prediction
        ewma_pred = self.ewma_state

        # 1d. SMA Prediction
        sma_pred = np.mean(hist_vec[:self.sma_period])

        # 1e. 2D Particle Filter Prediction (Kinematic Model)
        # Process noise: higher for level, lower for velocity (trends change slower than prices)
        level_noise = np.random.normal(0, np.sqrt(self.Q), self.num_particles)
        vel_noise = np.random.normal(0, np.sqrt(self.Q / 10), self.num_particles)
        
        # Next Level = Current Level + Current Velocity + Noise
        pred_level = self.particles[:, 0] + self.particles[:, 1] + level_noise
        # Next Velocity = Current Velocity + Noise
        pred_vel = self.particles[:, 1] + vel_noise
        
        pf_pred_particles = np.column_stack((pred_level, pred_vel))
        pf_pred = np.average(pf_pred_particles[:, 0], weights=self.particle_weights)

        # ==========================================
        # 2. ERROR EVALUATION (MSE)
        # ==========================================
        self.prediction_count += 1
        
        self.fractional_sse += (current_basis - frac_pred) ** 2
        self.standard_sse += (current_basis - std_pred) ** 2
        self.ewma_sse += (current_basis - ewma_pred) ** 2
        self.sma_sse += (current_basis - sma_pred) ** 2
        self.pf_sse += (current_basis - pf_pred) ** 2

        # ==========================================
        # 3. PLOTTING RESTORED
        # ==========================================
        self.Plot("Prediction Comparison", "Actual Basis", current_basis)
        self.Plot("Prediction Comparison", "Fractional Fused", frac_pred)
        self.Plot("Prediction Comparison", "Standard 1D KF", std_pred)
        self.Plot("Prediction Comparison", "2D Particle", pf_pred)

        fractional_mse = self.fractional_sse / self.prediction_count
        pf_mse = self.pf_sse / self.prediction_count
        
        self.Plot("MSE Comparison", "Fractional MSE", fractional_mse)
        self.Plot("MSE Comparison", "2D Particle MSE", pf_mse)
        
        winning_delta = self.deltas[np.argmax(self.weights)]
        self.Plot("Leaderboard", "Best Delta", winning_delta)

        # ==========================================
        # 4. STATE UPDATES (LEARNING FROM REALITY)
        # ==========================================
        
        # 4a. Fractional Updates
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

        # 4b. Standard KF Update
        std_p_pred = self.std_kf_covar + self.Q
        std_innovation = current_basis - std_pred
        std_gain = std_p_pred / (std_p_pred + dynamic_R)
        self.std_kf_state = std_pred + std_gain * std_innovation
        self.std_kf_covar = (1 - std_gain) * std_p_pred

        # 4c. EWMA Update
        self.ewma_state = (self.ewma_alpha * current_basis) + ((1 - self.ewma_alpha) * ewma_pred)

        # 4e. 2D Particle Filter Update (Resampling)
        # Score particles based on how close their predicted LEVEL is to the actual basis
        pf_likelihoods = (1 / np.sqrt(2 * np.pi * dynamic_R)) * np.exp(-0.5 * ((current_basis - pf_pred_particles[:, 0])**2) / dynamic_R)
        self.particle_weights *= pf_likelihoods
        self.particle_weights += 1e-300 
        self.particle_weights /= np.sum(self.particle_weights)

        # Systematic Resampling
        n_eff = 1.0 / np.sum(self.particle_weights**2)
        if n_eff < self.num_particles / 2.0:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.particle_weights)
            # Retain both level and velocity for the winning particles
            self.particles = pf_pred_particles[indices]
            self.particle_weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.particles = pf_pred_particles

        # ==========================================
        # 5. ADVANCE TIME & EXECUTION
        # ==========================================
        self.basis_history.Add(current_basis)

        std_dev = np.std(hist_vec)
        z = (current_basis - frac_pred) / std_dev if std_dev > 0 else 0
        self.Plot("Z-Score", "Value", z)

        if abs(z) < 0.5 and self.Portfolio.Invested:
            self.Liquidate()
            self.last_trade_time = datetime.min 
            return
            
        if (self.Time - self.last_trade_time).total_seconds() < 6 * 3600: return

        if z > 3 and not self.Portfolio.Invested:
            self.SetHoldings(self.future, -0.2)
            self.SetHoldings(self.spot, 0.2)
            self.last_trade_time = self.Time
        elif z < -3 and not self.Portfolio.Invested:
            self.SetHoldings(self.future, 0.2)
            self.SetHoldings(self.spot, -0.2)
            self.last_trade_time = self.Time

    def OnEndOfAlgorithm(self):
        mse_dict = {
            "Fractional Caterpillar": self.fractional_sse / self.prediction_count,
            "2D Particle Filter": self.pf_sse / self.prediction_count,
            "Standard 1D Kalman": self.standard_sse / self.prediction_count,
            "EWMA (alpha=0.05)": self.ewma_sse / self.prediction_count,
            "24H SMA Baseline": self.sma_sse / self.prediction_count
        }

        sorted_mse = sorted(mse_dict.items(), key=lambda item: item[1])

        self.Debug("=========================================")
        self.Debug("    ULTIMATE PREDICTION SCORECARD        ")
        self.Debug("=========================================")
        self.Debug(f"Total Predictions: {self.prediction_count}")
        self.Debug("-----------------------------------------")
        for rank, (name, mse) in enumerate(sorted_mse, 1):
            self.Debug(f"#{rank}. {name}: {mse:.10f}")
        self.Debug("=========================================")

        winner_name, winner_mse = sorted_mse[0]
        frac_mse = mse_dict["Fractional Caterpillar"]

        if winner_name == "Fractional Caterpillar":
            runner_up_mse = sorted_mse[1][1]
            imp = ((runner_up_mse - winner_mse) / runner_up_mse) * 100
            self.Debug(f" FRACTIONAL MODEL WINS! Beat runner-up by {imp:.2f}%")
        else:
            imp = ((frac_mse - winner_mse) / frac_mse) * 100
            self.Debug(f" Fractional Model lost to {winner_name} by {imp:.2f}%")
        self.Debug("=========================================")
