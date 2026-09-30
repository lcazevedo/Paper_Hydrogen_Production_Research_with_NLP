"""
Growth-model fitting library used to select and characterize each topic's
2014-2024 publication trend (SI Tables reporting the model-selection statistics
and the fitted parameters).

Fits five candidate functional forms (Linear, Exponential, Plateau, Logistic,
Gompertz) under two count distributions (Poisson, Negative Binomial NB2),
selects among them by corrected AIC (AICc), and reports Akaike weights,
McFadden's and Efron's pseudo-R^2, a Durbin-Watson autocorrelation statistic,
and a deviance goodness-of-fit test. See `reproduce_growth_model_fitting.py`
for a runnable reproduction against the topic-level annual counts already
deposited in this repository (`pipeline/relevance_index_input_data.csv`).

Note on the Gompertz/Logistic/Plateau asymptote bound: the lower bound on `A`
(and `L`) is fixed at the observed maximum count. For topics whose growth has
not yet visibly decelerated, this constraint is active (the fitted asymptote
equals the observed maximum exactly) -- a deliberate modeling choice (an
asymptote below the already-observed count is not physically meaningful for a
cumulative-style count that has not declined), not a fitting error. Freeing
this bound changes the AICc of the affected topics by at most a few points and
does not change the selected model family.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2

# -------------------------------------------------------------------------
# Growth Model Formulations
# -------------------------------------------------------------------------

def linear(x, a, b):
    """Linear growth model: y = a * x + b."""
    return np.maximum(a * x + b, 1e-10)

def exponential(x, A, k):
    """Exponential growth model: y = A * exp(k * x)."""
    return np.maximum(A * np.exp(k * x), 1e-10)

def plateau(x, A, k):
    """Plateau (asymptotic exponential) model: y = A * (1 - exp(-k * x))."""
    return np.maximum(A * (1.0 - np.exp(-k * x)), 1e-10)

def logistic(x, L, k, x0):
    """Logistic growth model: y = L / (1 + exp(-k * (x - x0)))."""
    return np.maximum(L / (1.0 + np.exp(-k * (x - x0))), 1e-10)

def gompertz(x, A, k, x0):
    """Gompertz growth model: y = A * exp(-exp(-k * (x - x0)))."""
    return np.maximum(A * np.exp(-np.exp(-k * (x - x0))), 1e-10)

# -------------------------------------------------------------------------
# Robust Initial Parameter Heuristics
# -------------------------------------------------------------------------

def get_initial_params(x, y, model_name):
    """
    Computes robust, data-driven initial parameter guesses for growth models 
    to guarantee numerical convergence of maximum likelihood estimation.
    """
    n = len(x)
    y_pos = np.maximum(y, 1.0)
    x_span = max(x) - min(x)
    if x_span == 0:
        x_span = 1.0

    if model_name == "Linear":
        slope, intercept = np.polyfit(x, y, 1)
        return [max(slope, 1e-5), max(intercept, 1e-5)]
        
    elif model_name == "Exponential":
        k, lnA = np.polyfit(x, np.log(y_pos), 1)
        # Clamp initial rate k to avoid extreme overflows
        return [max(np.exp(lnA), 1e-5), np.clip(k, 1e-5, 2.0)]
        
    elif model_name == "Plateau":
        A = 1.2 * max(y)
        mid_idx = n // 2
        y_mid = y_pos[mid_idx]
        ratio = 1.0 - y_mid / A
        ratio = np.clip(ratio, 0.01, 0.99)
        k = -np.log(ratio) / max(x[mid_idx], 1.0)
        return [A, max(k, 1e-5)]
        
    elif model_name == "Logistic":
        L = 1.2 * max(y)
        # Find x where y is closest to L/2
        idx = np.argmin(np.abs(y - L / 2.0))
        x0 = x[idx]
        # Standard growth rate starting guess
        k = 4.0 / x_span
        return [L, max(k, 1e-5), x0]
        
    elif model_name == "Gompertz":
        A = 1.2 * max(y)
        # Inflection point is at A/e
        idx = np.argmin(np.abs(y - A / np.e))
        x0 = x[idx]
        k = 2.0 / x_span
        return [A, max(k, 1e-5), x0]
        
    raise ValueError(f"Unknown model name: {model_name}")

# -------------------------------------------------------------------------
# Helper functions for parameter covariance
# -------------------------------------------------------------------------

def compute_hessian(fun, x0, eps=1e-5):
    """
    Computes the Hessian matrix of function `fun` at parameter vector `x0`
    using central finite differences with parameter-adaptive step sizes.
    """
    n = len(x0)
    hessian = np.zeros((n, n))
    
    # Scale finite-difference steps to parameter magnitude
    h_vec = np.array([eps * (abs(val) + 1.0) for val in x0])
    
    for i in range(n):
        for j in range(n):
            h_i = h_vec[i]
            h_j = h_vec[j]
            if i == j:
                p_plus = np.array(x0, dtype=float)
                p_minus = np.array(x0, dtype=float)
                p_plus[i] += h_i
                p_minus[i] -= h_i
                hessian[i, i] = (fun(p_plus) - 2.0 * fun(x0) + fun(p_minus)) / (h_i ** 2)
            else:
                p_pp = np.array(x0, dtype=float)
                p_pm = np.array(x0, dtype=float)
                p_mp = np.array(x0, dtype=float)
                p_mm = np.array(x0, dtype=float)
                
                p_pp[i] += h_i; p_pp[j] += h_j
                p_pm[i] += h_i; p_pm[j] -= h_j
                p_mp[i] -= h_i; p_mp[j] += h_j
                p_mm[i] -= h_i; p_mm[j] -= h_j
                
                hessian[i, j] = (fun(p_pp) - fun(p_pm) - fun(p_mp) + fun(p_mm)) / (4.0 * h_i * h_j)
    return hessian

# -------------------------------------------------------------------------
# Bibliometric Trend Analyzer Class
# -------------------------------------------------------------------------

class BibliometricTrendAnalyzer:
    """
    A statistically rigorous framework for fitting and comparing non-linear growth 
    models on publication count data. Supports Poisson and Negative Binomial (NB2) 
    maximum likelihood estimation, covariance-based confidence intervals, and residual diagnostics.
    """
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n_obs = len(y)
        
        # Model dictionary: name -> (function, parameter_names, bounds)
        self.models = {
            "Linear": (linear, ["a", "b"], [(0.0, None), (0.0, None)]),
            "Exponential": (exponential, ["A", "k"], [(1e-8, None), (0.0, None)]),
            "Plateau": (plateau, ["A", "k"], [(max(self.y), None), (0.0, None)]),
            "Logistic": (logistic, ["L", "k", "x0"], [(max(self.y), None), (0.0, None), (self.x.min() - 10, self.x.max() + 10)]),
            "Gompertz": (gompertz, ["A", "k", "x0"], [(max(self.y), None), (0.0, None), (self.x.min() - 10, self.x.max() + 10)])
        }
        
        self.results = {}
        
    def poisson_nll(self, params, model_func):
        """Negative log-likelihood of the Poisson regression model."""
        mu = model_func(self.x, *params)
        mu = np.maximum(mu, 1e-10)
        return -np.sum(self.y * np.log(mu) - mu - gammaln(self.y + 1.0))
        
    def nb2_nll(self, params, model_func):
        """Negative log-likelihood of the Negative Binomial (NB2) regression model."""
        model_params = params[:-1]
        alpha = np.maximum(params[-1], 1e-8)  # Overdispersion parameter clamped to prevent division by zero
        mu = model_func(self.x, *model_params)
        mu = np.maximum(mu, 1e-10)
        
        r = 1.0 / alpha
        term1 = gammaln(self.y + r)
        term2 = gammaln(self.y + 1.0)
        term3 = gammaln(r)
        term4 = self.y * np.log(alpha * mu)
        term5 = (self.y + r) * np.log(1.0 + alpha * mu)
        
        ll = term1 - term2 - term3 + term4 - term5
        return -np.sum(ll)

    def fit_null_model(self, dist_type="poisson"):
        """Fits an intercept-only null model to compute Null Log-Likelihood and Null Deviance."""
        mean_y = np.maximum(np.full_like(self.y, np.mean(self.y)), 1e-10)
        if dist_type == "poisson":
            # Return LL (log-likelihood, negative value) — consistent with NB2 branch below
            null_ll = np.sum(self.y * np.log(mean_y) - mean_y - gammaln(self.y + 1.0))
            y_pos = self.y > 0
            null_dev = 2.0 * np.sum(self.y[y_pos] * np.log(self.y[y_pos] / mean_y[y_pos])) - 2.0 * np.sum(self.y - mean_y)
            return null_ll, null_dev
        elif dist_type == "nb2":
            def null_nll(alpha_val):
                alpha = np.maximum(alpha_val[0], 1e-8)
                r = 1.0 / alpha
                term1 = gammaln(self.y + r)
                term2 = gammaln(self.y + 1.0)
                term3 = gammaln(r)
                term4 = self.y * np.log(alpha * mean_y)
                term5 = (self.y + r) * np.log(1.0 + alpha * mean_y)
                return -np.sum(term1 - term2 - term3 + term4 - term5)
                
            res = minimize(null_nll, [0.1], bounds=[(1e-5, 10.0)], method='L-BFGS-B')
            null_ll = -res.fun
            alpha_null = np.maximum(res.x[0], 1e-8)
            
            y_pos = self.y > 0
            term1 = np.zeros_like(self.y, dtype=float)
            term1[y_pos] = self.y[y_pos] * np.log((self.y[y_pos] / mean_y[y_pos]) * ((1.0 + alpha_null * mean_y[y_pos]) / (1.0 + alpha_null * self.y[y_pos])))
            term2 = (1.0 / alpha_null) * np.log((1.0 + alpha_null * mean_y) / (1.0 + alpha_null * self.y))
            null_dev = 2.0 * np.sum(term1 + term2)
            return null_ll, null_dev
        return np.nan, np.nan

    def fit(self, dist_type="poisson"):
        """Fits all growth models and calculates fit quality and information criteria."""
        dist_type = dist_type.lower()
        if dist_type not in ["poisson", "nb2"]:
            raise ValueError("dist_type must be either 'poisson' or 'nb2'")
            
        null_loglik, null_dev = self.fit_null_model(dist_type)
        out = []
        
        for name, (func, param_names, bounds) in self.models.items():
            init_guess = get_initial_params(self.x, self.y, name)
            
            if dist_type == "poisson":
                nll_func = lambda p: self.poisson_nll(p, func)
                res = minimize(nll_func, init_guess, bounds=bounds, method='L-BFGS-B')
                opt_params = res.x
                mle_ll = -res.fun
                k_params = len(opt_params)
            else:  # NB2
                nll_func = lambda p: self.nb2_nll(p, func)
                # Append alpha (dispersion) parameter
                init_guess_nb = list(init_guess) + [0.1]
                bounds_nb = list(bounds) + [(1e-5, 10.0)]
                res = minimize(nll_func, init_guess_nb, bounds=bounds_nb, method='L-BFGS-B')
                opt_params = res.x
                mle_ll = -res.fun
                k_params = len(opt_params) # includes alpha
                
            # Estimate Covariance Matrix (Hessian of NLL)
            cov, hessian = compute_hessian_and_cov(nll_func, opt_params)
            
            # Standard errors
            se = np.zeros_like(opt_params)
            for idx in range(len(opt_params)):
                if cov[idx, idx] > 0:
                    se[idx] = np.sqrt(cov[idx, idx])
                else:
                    se[idx] = np.nan
                    
            # Goodness-of-Fit
            mu_pred = func(self.x, *opt_params[:-1]) if dist_type == "nb2" else func(self.x, *opt_params)
            mu_pred = np.maximum(mu_pred, 1e-10)
            
            # Deviance and Chi-Squared lack-of-fit test
            if dist_type == "poisson":
                dev = 2.0 * np.sum(self.y[self.y > 0] * np.log(self.y[self.y > 0] / mu_pred[self.y > 0])) - 2.0 * np.sum(self.y - mu_pred)
            else:
                alpha = np.maximum(opt_params[-1], 1e-8)
                y_pos = self.y > 0
                term1 = np.zeros_like(self.y, dtype=float)
                term1[y_pos] = self.y[y_pos] * np.log((self.y[y_pos] / mu_pred[y_pos]) * ((1.0 + alpha * mu_pred[y_pos]) / (1.0 + alpha * self.y[y_pos])))
                term2 = (1.0 / alpha) * np.log((1.0 + alpha * mu_pred) / (1.0 + alpha * self.y))
                dev = 2.0 * np.sum(term1 + term2)

            # True McFadden R2 = 1 - LL_fitted / LL_null  (log-likelihood ratio index)
            # null_loglik is LL_null (negative); mle_ll is LL_fitted (negative, larger = better)
            mcfadden_r2 = max(0.0, 1.0 - (mle_ll / null_loglik)) if (null_loglik != 0 and np.isfinite(null_loglik)) else np.nan
            # Deviance reduction R2 = 1 - D_fitted / D_null  (explained deviance index)
            deviance_r2 = max(0.0, 1.0 - (dev / null_dev)) if null_dev > 0 else np.nan
            # Efron's R2 (OLS-analogue on the response scale)
            efron_r2 = max(0.0, 1.0 - (np.sum((self.y - mu_pred)**2) / np.sum((self.y - np.mean(self.y))**2)))
                
            df_dev = self.n_obs - k_params
            p_val_dev = chi2.sf(dev, df_dev) if df_dev > 0 else np.nan
            
            # Pearson residuals and Durbin-Watson statistic
            if dist_type == "poisson":
                var_y = mu_pred
            else:
                alpha = opt_params[-1]
                var_y = mu_pred + alpha * (mu_pred**2)
            pearson_res = (self.y - mu_pred) / np.sqrt(var_y)
            
            # Durbin-Watson Autocorrelation check
            dw_stat = np.sum(np.diff(pearson_res)**2) / np.sum(pearson_res**2)
            
            # Standard information criteria
            aic = 2.0 * k_params - 2.0 * mle_ll
            aicc = aic + (2.0 * k_params * (k_params + 1.0)) / (self.n_obs - k_params - 1.0) if self.n_obs > k_params + 1 else np.nan
            bic = k_params * np.log(self.n_obs) - 2.0 * mle_ll
            
            out.append({
                "Model": name,
                "Params": opt_params,
                "SE": se,
                "logLik": mle_ll,
                "AIC": aic,
                "AICc": aicc,
                "BIC": bic,
                "McFadden_R2": mcfadden_r2,
                "DevR2": deviance_r2,
                "Efron_R2": efron_r2,
                "Deviance": dev,
                "Deviance_pVal": p_val_dev,
                "DW_Stat": dw_stat,
                "Covariance": cov,
                "PearsonResiduals": pearson_res
            })
            
        # Compile dataframe
        df_res = pd.DataFrame(out).sort_values("AICc").reset_index(drop=True)
        
        # Calculate Delta AICc and Akaike weights
        df_res["DeltaAICc"] = df_res.AICc - df_res.AICc.min()
        w = np.exp(-0.5 * df_res.DeltaAICc)
        df_res["Weight"] = w / w.sum()
        df_res["EvidenceRatio"] = df_res.Weight.max() / df_res.Weight
        
        # Delta AIC and Delta BIC relative to the best model in each criterion
        df_res["DeltaAIC"] = df_res.AIC - df_res.AIC.min()
        df_res["DeltaBIC"] = df_res.BIC - df_res.BIC.min()
        
        # Save results internally
        self.results[dist_type] = df_res
        return df_res

    def run_dispersion_analysis(self):
        """
        Performs a Poisson overdispersion analysis, calculating Pearson's dispersion ratio 
        and performing a Likelihood Ratio Test (LRT) between Poisson and NB2 variants.
        """
        poisson_res = self.fit("poisson")
        nb_res = self.fit("nb2")
        
        disp_data = []
        for name in self.models.keys():
            # Get Poisson fit
            p_row = poisson_res[poisson_res.Model == name].iloc[0]
            nb_row = nb_res[nb_res.Model == name].iloc[0]
            
            # Pearson dispersion ratio = Chi-squared Pearson / df
            p_params_len = len(p_row.Params)
            df_poisson = self.n_obs - p_params_len
            mu_pred = self.models[name][0](self.x, *p_row.Params)
            chi2_pearson = np.sum(((self.y - mu_pred)**2) / np.maximum(mu_pred, 1e-10))
            disp_ratio = chi2_pearson / df_poisson if df_poisson > 0 else np.nan
            
            # Likelihood Ratio Test: Poisson (restricted) vs NB2 (unrestricted)
            lrt_stat = 2.0 * (nb_row.logLik - p_row.logLik)
            # LRT is mixture of 0 and chi2_1 (p-value = 0.5 * chi2.sf(lrt, 1))
            lrt_pval = 0.5 * chi2.sf(max(lrt_stat, 0.0), 1)
            
            disp_data.append({
                "Model": name,
                "Poisson_logLik": p_row.logLik,
                "NB2_logLik": nb_row.logLik,
                "LRT_Stat": lrt_stat,
                "LRT_pVal": lrt_pval,
                "Pearson_Dispersion": disp_ratio,
                "Recommendation": "Negative Binomial (NB2)" if lrt_pval < 0.05 or disp_ratio > 1.5 else "Poisson"
            })
            
        return pd.DataFrame(disp_data)

    def compute_rmax(self, model_name, params, cov=None):
        """
        Computes the maximum annual growth rate (R_max in publications/year) 
        and its standard error via the Delta Method.
        
        Mathematical Formulations:
        --------------------------
        - Logistic: R_max = (k * L) / 4   (at inflection point t_0)
        - Gompertz: R_max = (k * A) / e   (at inflection point t_0)
        - Linear:   R_max = a            (constant growth rate)
        - Plateau:  R_max = A * k        (initial maximum growth rate at t_min)
        - Exponential: R_max = k * A * exp(k * t_max) (maximum observed rate at t_max)
        """
        params = np.asarray(params, dtype=float)
        rmax_se = np.nan
        
        if model_name == "Logistic":
            L, k, x0 = params[0], params[1], params[2]
            rmax = (L * k) / 4.0
            t_rmax = x0
            if cov is not None and cov.shape[0] >= 3:
                grad = np.array([k / 4.0, L / 4.0, 0.0])
                var_rmax = grad.T @ cov[:3, :3] @ grad
                if var_rmax > 0:
                    rmax_se = np.sqrt(var_rmax)

        elif model_name == "Gompertz":
            A, k, x0 = params[0], params[1], params[2]
            rmax = (A * k) / np.e
            t_rmax = x0
            if cov is not None and cov.shape[0] >= 3:
                grad = np.array([k / np.e, A / np.e, 0.0])
                var_rmax = grad.T @ cov[:3, :3] @ grad
                if var_rmax > 0:
                    rmax_se = np.sqrt(var_rmax)

        elif model_name == "Linear":
            a, b = params[0], params[1]
            rmax = a
            t_rmax = np.nan
            if cov is not None and cov.shape[0] >= 2:
                rmax_se = np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.nan

        elif model_name == "Plateau":
            A, k = params[0], params[1]
            rmax = A * k
            t_rmax = self.x.min()
            if cov is not None and cov.shape[0] >= 2:
                grad = np.array([k, A])
                var_rmax = grad.T @ cov[:2, :2] @ grad
                if var_rmax > 0:
                    rmax_se = np.sqrt(var_rmax)

        elif model_name == "Exponential":
            A, k = params[0], params[1]
            t_max = self.x.max()
            rmax = k * A * np.exp(k * t_max)
            t_rmax = t_max
            if cov is not None and cov.shape[0] >= 2:
                grad = np.array([k * np.exp(k * t_max), A * np.exp(k * t_max) * (1.0 + k * t_max)])
                var_rmax = grad.T @ cov[:2, :2] @ grad
                if var_rmax > 0:
                    rmax_se = np.sqrt(var_rmax)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return rmax, t_rmax, rmax_se

    def print_parameter_table(self, model_name, dist_type="poisson"):
        """Prints a detailed parameter table including R_max with standard errors and 95% confidence intervals."""
        dist_type = dist_type.lower()
        if dist_type not in self.results:
            self.fit(dist_type)
            
        df = self.results[dist_type]
        row = df[df.Model == model_name].iloc[0]
        params = row.Params
        se = row.SE
        cov = row.Covariance
        
        # Get parameter names
        names = list(self.models[model_name][1])
        if dist_type == "nb2":
            names.append("alpha (dispersion)")
            
        param_table = []
        for name_val, val, err in zip(names, params, se):
            ci_lower = val - 1.96 * err if not np.isnan(err) else np.nan
            ci_upper = val + 1.96 * err if not np.isnan(err) else np.nan
            param_table.append({
                "Parameter": name_val,
                "Estimate": val,
                "StdError": err,
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper
            })
            
        # Calculate R_max for this model
        rmax, t_rmax, rmax_se = self.compute_rmax(model_name, params[:-1] if dist_type == "nb2" else params, cov[:-1, :-1] if dist_type == "nb2" else cov)
        ci_lower_r = rmax - 1.96 * rmax_se if not np.isnan(rmax_se) else np.nan
        ci_upper_r = rmax + 1.96 * rmax_se if not np.isnan(rmax_se) else np.nan
        
        param_table.append({
            "Parameter": "R_max (max growth rate)",
            "Estimate": rmax,
            "StdError": rmax_se,
            "95% CI Lower": ci_lower_r,
            "95% CI Upper": ci_upper_r
        })
            
        return pd.DataFrame(param_table)

    def plot(self, dist_type="poisson", show_ci=True):
        """
        Renders a publication-ready plot showing the data points, fitted growth models, 
        and 95% parameter uncertainty bands (Delta-method prediction bands).
        """
        dist_type = dist_type.lower()
        if dist_type not in self.results:
            self.fit(dist_type)
            
        df_res = self.results[dist_type]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.x, self.y, color='black', edgecolor='none', s=35, label="Observed Data", zorder=3)
        
        # Dense grid for plotting smooth lines
        xx = np.linspace(self.x.min(), self.x.max(), 300)
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        
        for idx, (_, row) in enumerate(df_res.iterrows()):
            model_name = row.Model
            func = self.models[model_name][0]
            
            # Predict values
            params = row.Params[:-1] if dist_type == "nb2" else row.Params
            cov = row.Covariance[:-1, :-1] if dist_type == "nb2" else row.Covariance
            
            yy = func(xx, *params)
            
            # Plot fitted curve
            color = colors[idx % len(colors)]
            line = ax.plot(xx, yy, label=f"{model_name} (w={row.Weight:.3f})", color=color, linewidth=2)
            
            if show_ci:
                # Delta method prediction variance
                pred_var = self._compute_prediction_variance(model_name, params, cov, xx)
                pred_se = np.sqrt(pred_var)
                
                # Under normal approximation, 95% CI bounds
                ci_lower = np.maximum(yy - 1.96 * pred_se, 1e-10)
                ci_upper = yy + 1.96 * pred_se
                
                ax.fill_between(xx, ci_lower, ci_upper, color=color, alpha=0.1)
                
        ax.set_xlabel("Time (Years relative)", fontsize=11)
        ax.set_ylabel("Publication Count", fontsize=11)
        ax.set_title(f"Model Fits with 95% Delta-Method Confidence Bands ({dist_type.upper()})", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        return fig

    def _compute_prediction_variance(self, model_name, params, cov, xx, eps=1e-5):
        """Calculates prediction variance via the Delta Method."""
        func = self.models[model_name][0]
        n_params = len(params)
        n_points = len(xx)
        pred_var = np.zeros(n_points)
        
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return pred_var
            
        for idx, x_val in enumerate(xx):
            grad = np.zeros(n_params)
            for i in range(n_params):
                p_plus = np.array(params, dtype=float)
                p_minus = np.array(params, dtype=float)
                p_plus[i] += eps
                p_minus[i] -= eps
                grad[i] = (func(x_val, *p_plus) - func(x_val, *p_minus)) / (2.0 * eps)
            pred_var[idx] = grad.T @ cov @ grad
            
        return pred_var

# -------------------------------------------------------------------------
# Helper function for parameter covariance
# -------------------------------------------------------------------------
def compute_hessian_and_cov(nll_func, opt_params):
    """
    Computes the parameter covariance matrix by inverting the Hessian
    of the negative log-likelihood with positive-definite eigenvalue projection.
    """
    hessian = compute_hessian(nll_func, opt_params)
    try:
        cov = np.linalg.inv(hessian)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)) or np.any(np.diag(cov) <= 0):
            # Eigenvalue regularization to guarantee positive-definiteness
            eigvals, eigvecs = np.linalg.eigh((hessian + hessian.T) / 2.0)
            eigvals = np.maximum(eigvals, 1e-6)
            cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh((hessian + hessian.T) / 2.0)
        eigvals = np.maximum(eigvals, 1e-6)
        cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    return cov, hessian

# -------------------------------------------------------------------------
# Batch Multi-Topic Trend Analyzer Class
# -------------------------------------------------------------------------

class BatchTrendAnalyzer:
    """
    Automates multi-topic bibliometric growth modeling across a tabular dataset 
    where rows represent research topics and columns represent annual publication counts.
    """
    def __init__(self, data, topic_col, year_cols):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing topics/keywords and annual publication counts.
        topic_col : str
            Name of the column containing topic labels.
        year_cols : list or array-like
            Column names corresponding to annual publication counts (e.g. ['2014', '2015', ...]).
        """
        self.data = data.copy()
        self.topic_col = topic_col
        self.year_cols = [str(y) for y in year_cols]
        # Parse years as numeric time sequence
        self.years = np.array([int(y) for y in year_cols])
        self.t = self.years - self.years.min()
        
    def analyze_all(self, auto_dist=True, save_plots_dir=None, show_plots=False, **kwargs):
        """
        Runs the growth model comparison for all topics in the dataset.
        
        Parameters:
        -----------
        auto_dist : bool (default=True)
            If True, uses Likelihood Ratio Tests (LRT) to automatically select 
            between Poisson and Negative Binomial (NB2) distributions per topic.
        save_plots_dir : str or None (default=None)
            If specified, saves individual PNG plot figures of the model fits for each topic.
        show_plots : bool (default=False)
            If True, renders the plot figure for each topic directly in the notebook output.
            
        Returns:
        --------
        pd.DataFrame : Master summary table containing growth classifications, best models, 
                       fit quality metrics, and parameter estimates with standard errors.
        """
        results_list = []
        
        if save_plots_dir:
            import os
            os.makedirs(save_plots_dir, exist_ok=True)
            
        for idx, row in self.data.iterrows():
            topic_name = str(row[self.topic_col])
            y_counts = row[self.year_cols].values.astype(float)
            
            analyzer = BibliometricTrendAnalyzer(self.t, y_counts)
            disp_df = analyzer.run_dispersion_analysis()
            
            p_res = analyzer.fit("poisson")
            nb_res = analyzer.fit("nb2")
            
            if auto_dist:
                top_p_model = p_res.iloc[0].Model
                lrt_row = disp_df[disp_df.Model == top_p_model].iloc[0]
                
                # Check if NB2 is recommended due to overdispersion or LRT significance
                if lrt_row.LRT_pVal < 0.05 or lrt_row.Pearson_Dispersion > 1.5:
                    chosen_dist = "nb2"
                    best_df = nb_res
                else:
                    chosen_dist = "poisson"
                    best_df = p_res
            else:
                chosen_dist = "poisson"
                best_df = p_res
                
            best_row = best_df.iloc[0]
            best_model = best_row.Model
            
            # Format parameters into a readable string
            param_df = analyzer.print_parameter_table(best_model, chosen_dist)
            param_parts = []
            for _, r in param_df.iterrows():
                if not np.isnan(r.StdError):
                    param_parts.append(f"{r.Parameter}={r.Estimate:.3f} (±{r.StdError:.3f})")
                else:
                    param_parts.append(f"{r.Parameter}={r.Estimate:.3f}")
            param_str = ", ".join(param_parts)
            
            # Growth Pattern Classification
            if best_model == "Exponential":
                category = "Exponential Growth (Accelerating)"
            elif best_model == "Plateau":
                category = "Plateau Growth (Decelerating/Saturation)"
            elif best_model == "Linear":
                category = "Linear Growth (Constant Rate)"
            elif best_model in ["Logistic", "Gompertz"]:
                category = f"{best_model} Growth (Sigmoidal Saturation)"
            else:
                category = best_model
                
            total_pubs = int(np.sum(y_counts))
            
            # Compute R_max for the best model
            best_params = best_row.Params[:-1] if chosen_dist == "nb2" else best_row.Params
            best_cov = best_row.Covariance[:-1, :-1] if chosen_dist == "nb2" else best_row.Covariance
            rmax, t_rmax, rmax_se = analyzer.compute_rmax(best_model, best_params, best_cov)
            
            results_list.append({
                "Topic": topic_name,
                "Total_Publications": total_pubs,
                "Growth_Category": category,
                "Best_Model": best_model,
                "Distribution_Used": chosen_dist.upper(),
                "R_max": rmax,
                "R_max_SE": rmax_se,
                "Akaike_Weight": best_row.Weight,
                "Evidence_Ratio": best_row.EvidenceRatio,
                "AIC": best_row.AIC,
                "DeltaAIC": best_row.DeltaAIC,
                "AICc": best_row.AICc,
                "DeltaAICc": best_row.DeltaAICc,
                "BIC": best_row.BIC,
                "DeltaBIC": best_row.DeltaBIC,
                "McFadden_R2": best_row.McFadden_R2,
                "DevR2": best_row.DevR2,
                "Efron_R2": best_row.Efron_R2,
                "Durbin_Watson": best_row.DW_Stat,
                "Deviance_pVal": best_row.Deviance_pVal,
                "Model_Parameters": param_str
            })
            
            # Save and/or show plot figure
            if save_plots_dir or show_plots:
                fig = analyzer.plot(chosen_dist, show_ci=True)
                fig.suptitle(f"Topic: {topic_name} — Best Model: {best_model} ({chosen_dist.upper()})", fontsize=11, fontweight='bold')
                plt.tight_layout()
                
                if save_plots_dir:
                    import os
                    safe_filename = "".join([c if c.isalnum() else "_" for c in topic_name])
                    fig.savefig(os.path.join(save_plots_dir, f"{safe_filename}_trend.png"), dpi=200)
                    
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)
                
        summary_df = pd.DataFrame(results_list)
        return summary_df

