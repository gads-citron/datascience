# season lengths -> len = m
season_len = 24*7
# trend extraction regularization factors -> len = 2
trend_regs = [1, 0.1]
# season regularization factors -> len = 3 * m
season_regs = [[1, 1, 1]]  # lambda 1, 2, 3 for seasonality  2
# weighting factor for each non-local seasonal filter (Eq. 10) -> len = m
alpha = 0.5
# normalization factor (Eq. 10)
z = 1
# denoising weights -> len = 2
denoise_ds = [4.0, 1.0]
# season decomposition weights -> len = 2
season_ds = [5.0, 1.0]
K = 3
H = 2