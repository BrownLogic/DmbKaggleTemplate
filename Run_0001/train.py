from kaggle import settings
import pandas as pd
import logging.config
import json
from persistance.execution_context import PersistModel, FileObjectType
from sklearn.pipeline import FeatureUnion, Pipeline
from data_preparation.transformers import ColumnExtractor, LetterCountTransformer, NaNCountTransformer, \
    NanToZeroTransformer, MultiColumnLabelEncoder, LetterExtractionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

the_settings = settings.Settings()
"""
One goal I have is to create a 'record' of every attempt in the database
It's kinda like logging, but I want to create a 'run_id' along with logs
"""

def do_the_deal():
    """
    Drives the whole process
    """
    # Stage
    model_persistor = PersistModel(the_settings.saved_object_directory)
    try:
        logging.info("Beginning {}".format(model_persistor.get_log_context()))
        train_data = get_train_data()
        test_data = get_test_data()
        train_data_features, test_data_features = extract_features(train_data, test_data, model_persistor)
        # In this case, we included test_data_features so that we could encode
        # as many features as we could know about

        # Use train data for classifier.
        X_train = train_data_features
        y_train = train_data[the_settings.target]

        model_persistor.add_object_to_save(X_train, FileObjectType.train_feature)
        model_persistor.add_object_to_save(y_train, FileObjectType.train_target)
        model = get_model()
        score = score_the_model(model, X_train, y_train,model_persistor)

        #do final fit with all of the data
        model.fit(X_train, y_train)
        model_persistor.add_object_to_save(model, FileObjectType.predictor_model)
        model_persistor.add_note("score = {}".format(score))
        logging.info(score)
        logging.info("Ending {}".format(model_persistor.get_log_context()))
    except BaseException as e:
        model_persistor.add_note("There was an error.  Check the logs for more details. {}", str(e))
        logging.exception("Unexpected Error")
    finally: # I may should pull this out of the handler.
        model_persistor.save_all()

def get_test_data():
    """
    get the test data and do any simple pre processing

    @:returns pandas data frame
    """
    return pd.read_csv(the_settings.test_file_path)


def get_train_data():
    """
    get the train data and do any simple pre processing

    @:returns pandas data frame
    """
    return pd.read_csv(the_settings.train_file_path)


def extract_features(train_data, test_data=None, model_persistor=None):
    """
    Does feature enrichment and enhancement (separate from modeling)
    if test_data is passed, it is included in the processing and split out again
    this is to account for encoding where feature categories aren't in the train set
    :param train_data: Training data
    :param test_data:  Test data
    :model_persistor: An instance of PersistModel.  When passed, supporting objects can be added
    """
    if test_data is not None:
        data_to_process = pd.concat([train_data[the_settings.all_features], test_data[the_settings.all_features]], ignore_index=True)
    else:
        data_to_process = train_data


    feature_extraction = FeatureUnion(
        [('numeric_features_standardized', Pipeline([
            ('numeric_features_raw', FeatureUnion([
                ('numeric_features', ColumnExtractor(the_settings.numeric_features))
                , ('v22_letter_count', LetterCountTransformer(the_settings.special_string_features))
                , ('Nan_count', NaNCountTransformer())
                ]))
            , ('zero_na', NanToZeroTransformer())
            ]))
            , ('string_features_standardized', Pipeline([
            ('string_features', ColumnExtractor(the_settings.string_features))
            , ('label', MultiColumnLabelEncoder())
            # returns a sparse array...the exact kind we need
            , ('one_hot', OneHotEncoder(sparse=True))
                ]))
            , ('v22_standardized', Pipeline([
            ('extract', LetterExtractionTransformer(the_settings.special_string_features))
            , ('label', MultiColumnLabelEncoder())
            , ('one_hot', OneHotEncoder(sparse=True))
        ]))

    ])

    fitted_feature_model = feature_extraction.fit(data_to_process)
    model_persistor.add_object_to_save(fitted_feature_model, FileObjectType.feature_model)
    extracted_features = fitted_feature_model.transform(data_to_process)

    return extracted_features[:len(train_data)], extracted_features[len(train_data):]


def get_model():
    """
    Returns a model along with any preparation
    """
    return LogisticRegression(random_state=1)


def score_the_model(model, X, y, model_persistor=None):
    scoring_type = 'log_loss'
    scores = cross_validation.cross_val_score(model, X, y, cv=5, scoring=scoring_type)
    model_persistor.add_score(scoring_type, -scores.mean())

    return "Accuracy: {:.2f} (+/- {:.2f})".format(-scores.mean(), scores.std())


if __name__ == '__main__':
    config = json.load(open(the_settings.logging_config, 'r'))
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    try:
        do_the_deal()
    except BaseException as e:
        logging.exception("Unexpected Error")


