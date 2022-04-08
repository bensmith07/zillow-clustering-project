def display_model_results(model_results, as_std_from_baseline=False):
    '''
    This function takes in the model_results dataframe created in the Model stage of the 
    project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of
    models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index grouped by metric_type then sample_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    if as_std_from_baseline:
        return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='std_from_baseline',
                                     aggfunc=lambda x: x)
    else:
        return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)        

def get_best_model_results(model_results, n_models=3):
    '''
    This function takes in the model_results dataframe created in the Modeling stage of the 
    project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)

    The function identifies the {n_models} models with the highest scores for the given metric
    type, as measured on the validate sample.

    It returns a dataframe of information about those models' performance in the tidy data format
    (as described above). 

    The resulting dataframe can be fed into the display_model_results function for convenient display formatting.
    '''
    # create an array of model numbers for the best performing models
    # by filtering the model_results dataframe for only validate scores
    best_models = (model_results[(model_results.sample_type == 'validate')]
                                                 # sort by score value in ascending order
                                                 .sort_values(by='score', 
                                                              ascending=True)
                                                 # take only the model number for the top n_models
                                                 .head(n_models).model_number
                                                 # and take only the values from the resulting dataframe as an array
                                                 .values)
    # create a dataframe of model_results for the models identified above
    # by filtering the model_results dataframe for only the model_numbers in the best_models array
    # TODO: make this so that it will return n_models, rather than only 3 models
    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results

