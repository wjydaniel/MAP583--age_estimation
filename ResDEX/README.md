# Process

* First use the 'apparent_age' to train a DEX model : model1

* Then use 'real_age - model1(x)' as label to train a residual model : model2

* In validate, we implement model1(x) + model2(x) to calculate MAE
