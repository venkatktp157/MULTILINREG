# JMP-Inspired Stepwise Regression Control Panel (Streamlit Clone)

An interactive, educational web application built with Python and Streamlit that mirrors the look, feel, and statistical output of SAS JMP's multi-linear stepwise regression platform. 

This tool is designed for teaching and demonstrating how individual features affect variance partitions ($SSE$, $SST$), structural metrics ($AIC$, $BIC$), and parameter constraints (Mallows' $C_p$) in real-time.

## 🚀 Features
- **Dual-Column Layout:** Live data viewer on the left; regression diagnostics control deck on the right.
- **Dynamic Parameter Selection:** Toggle variables manually using checkboxes to see diagnostics update instantly.
- **Classic JMP Diagnostics Panel:** Computes and displays $SSE$, $DFE$, $RMSE$, $R^2$, Adjusted $R^2$, Mallows' $C_p$, $AIC$, and $BIC$.
- **Estimates Matrix:** Evaluates individual parameter coefficients, standard errors, $t$-ratios, and exact $P$-values using `statsmodels`.

## 🛠️ Installation & Setup

1. **Clone or download** this repository to your local machine.
2. Ensure you have **Python 3.9+** installed.
3. Open your terminal in the project directory and install the necessary dependencies:
   ```bash
   pip install -r requirements.txt