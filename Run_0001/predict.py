from kaggle import settings
import pandas as pd
import logging.config
import json
from persistance.execution_context import PersistModel, FileObjectType

the_settings = settings.Settings()
"""
One goal I have is to create a 'record' of every attempt in the database
It's kinda like logging, but I want to create a 'run_id' along with logs
"""
def do_the_deal(run_id):
    """
    Drives the whole process
    """
    #  Starts with a run_id.  Assumes that there is a model already saved.
    model_persistor = PersistModel( the_settings.saved_object_directory, run_id = run_id, project_name=the_settings.competiton_name)

    logging.info("Beginning {}".format(model_persistor.get_log_context()))

    #  load the test data
    test_data = get_test_data()

    #  Load the feature transformation class
    feature_model = model_persistor.get_object(FileObjectType.feature_model)

    #  Transform the test data
    X_test = feature_model.transform(test_data)

    #  Load the model
    predictor_model = model_persistor.get_object(FileObjectType.predictor_model)

    #  Predict the target values
    y_predicted = predictor_model.predict_proba(X_test)

    #  Align the the predicted values with ID
    y_final = pd.DataFrame(test_data[the_settings.record_pk])
    y_final['PredictedProb']=[row[1] for row in y_predicted]


    submission_file_name = model_persistor.save_submission(y_final, the_settings.submissions_directory)
    logging.info("Finished {}".format(model_persistor.get_log_context()))
    logging.info("Saved file as {}".format(submission_file_name))


def get_test_data():
    """
    get the test data and do any simple pre processing

    @:returns pandas data frame
    """
    return pd.read_csv(the_settings.test_file_path)



if __name__ == '__main__':
    config = json.load(open(the_settings.logging_config, 'r'))
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    try:
        do_the_deal(139)
    except BaseException as e:
        logging.exception("Unexpected Error")

