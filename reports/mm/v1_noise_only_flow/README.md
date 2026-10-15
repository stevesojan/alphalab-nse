# Archived: first ablation with a noise-only order flow

This run used the flow WITHOUT a latent value or informed traders. Market makers that fade price moves
(contrarian EMA, RSI, support/resistance reversion, faded breakouts) showed Sharpe ratios of 250-350 with
zero drawdown. That is a simulator artifact: with only uninformed flow the mid price merely bounces, so
fading every move is nearly riskless. It was kept as evidence of why the informed-flow model was added
(see src/alphalab/lob/spec.py, 'informed flow'). Do not quote these numbers as results.
