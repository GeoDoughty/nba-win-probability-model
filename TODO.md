## Short Term

- [x] Figure out why model is predicting 0.58 for every game (is it) - This was due to too many features
- [x] Set up basic evals: % accuracy and win prob representation (did 70% of the 70% games win)
- [x] Clean up code so it's clear what's test and train
- [ ] Add bonus/fouls as feature

Check this out for help: <https://www.digitalocean.com/community/tutorials/logistic-regression-with-scikit-learn>

## medium term

### Features

- [ ] Add % instead of makes
- [ ] Add deviation from season averages
- [ ] Add bonus/fouls as feature
- [ ] Add interaction between home and away team

### Models

- [x] use lasso on the features for logit
  - [x] Normalise the data
  - [x] Fix RFECV
  - [x] try and asses accuracy and feature importance (SHAP?)
- [ ] XGBoost
- [ ] Bayesian ;)
- [ ] Analyse probability jumps
- [ ] Optuna for hyperparam optimisation
- [ ]

### Ideas

- Bet I could extract out the ball location at a point in time with tracking data
