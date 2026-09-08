# Install once: python -m pip install imvpy
import numpy as np
from imvpy import vanilla_imv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Simulate a binary outcome from one predictor.
rng = np.random.default_rng(42)
x = rng.normal(size=(1000, 1))
p = 1 / (1 + np.exp(-(2 * x[:, 0] - 1)))
y = rng.binomial(1, p)

# 2. Fit on training data and predict test outcomes.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
model = LogisticRegression().fit(x_train, y_train)
p_enhanced = model.predict_proba(x_test)[:, 1]

# 3. Compare against the training prevalence.
p_baseline = y_train.mean()
imv = vanilla_imv(p_baseline, p_enhanced, y_test)
print(f"IMV: {imv:.3f}")
