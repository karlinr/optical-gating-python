# DEFINITELY
- Implement drift corrector and consider how we can handle drift correction with the MLE estimator. Consider whether we need to have multiple different drift correctors.
- Add support for multiple timing boxes for triggering using the different approaches.
# PROBABLY
- Consider whether we can handle the barrier frame logic slightly cleaner, or perhaps combine the predictors and estimators together. The current approach feels messy as currently every estimator requires an output of the barrier frame, which feels unneccesary.
- Add a new predictor based on the MLE method which handles long-term updates
- Add a new predictor based on the PCA-MLE approach
# MAYBE
- Add support for the stage controller
