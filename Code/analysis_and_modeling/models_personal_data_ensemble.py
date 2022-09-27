import argparse

from apitep_utils import ArgumentParserHelper
from apitep_utils.analysis_modelling import AnalysisModeling
import logging
import numpy as np
import keys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as sklm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

log = logging.getLogger(__name__)


class QuadrimestersEnsemble(AnalysisModeling):
    quadrimester: int = None
    course: int = None
    final_analys_record_personal_access: pd.DataFrame = None
    model_number: str = None
    x_train: pd.DataFrame = None
    x_test: pd.DataFrame = None
    x_train_norm: pd.DataFrame = None
    x_test_norm: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None
    y_pred: pd.Series = None
    threshold: float = 0.47

    def get_best_hyperparameters_RandomForest(self):
        grid_params = {'max_depth': [8, 12],
                       'n_estimators': [50, 100, 200]}
        gs_RndForest = GridSearchCV(
            RandomForestClassifier(random_state=123),
            grid_params,
            scoring='recall',
            n_jobs=-1,
            cv=4
        )
        gs_RndForest.fit(self.x_train, self.y_train)
        return gs_RndForest.best_estimator_

    def get_best_hyperparameters_SVM(self):
        grid_params = {'C': [0.01, 0.1, 1, 10, 50, 100, 200],
                       'gamma': [0.001, 0.01, 0.1, 1, 10]}

        gs_SVM = GridSearchCV(
            SVC(random_state=123),
            grid_params,
            scoring='accuracy',
            n_jobs=-1,
            cv=4
        )
        gs_SVM.fit(self.x_train_norm, self.y_train)
        return gs_SVM.best_estimator_

    def get_best_hyperparameters_MLP(self):
        grid_params = {
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive']
        }
        gs_MLP = GridSearchCV(
            MLPClassifier(random_state=123, max_iter=1000),
            grid_params,
            scoring='accuracy',
            n_jobs=-1,
            cv=4
        )
        gs_MLP.fit(self.x_train, self.y_train)
        return gs_MLP.best_estimator_

    def parse_arguments(self):
        """
        Parse arguments provided via command line, and check if they are valid
        or not. Adequate defaults are provided when possible.

        Parsed arguments are:
        - paths to the input CSV datasets, separated with spaces.
        - path to the output CSV dataset.
        """

        log.info("Get integration arguments")
        log.debug("QuadrimestersEnsemble.parse_arguments()")

        argument_parser = argparse.ArgumentParser(description=self.description)
        argument_parser.add_argument("-i", "--input_path", required=True,
                                     help="path to the input CSV dataset")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        argument_parser.add_argument("-c", "--course", required=True,
                                     help="course to analyze")
        argument_parser.add_argument("-q", "--quadrimester", required=True,
                                     help="quadrimester of course to analyze")
        argument_parser.add_argument("-mn", "--model_number", required=True,
                                     help="number of model to create")

        arguments = argument_parser.parse_args()
        self.input_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.input_path)
        self.output_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.output_path,
            check_is_file=False)
        self.course = int(arguments.course)
        self.quadrimester = int(arguments.quadrimester)
        self.model_number = arguments.model_number

    def process(self):
        """
        Analysis of final_analys_record_personal_access
        """
        if self.course == 0:
            log.info("Analysis of analys_record_personal_access data")
        else:
            log.info("Analysis of analys_record_personal_access data"
                     " with data of course " + str(self.course) + " and quadrimester " + str(self.quadrimester))

        log.debug("QuadrimestersEnsemble.process()")

        note_bcket_array = np.array([5, 6.5, 8, 9.5, 10, 11.5, 13, 14])
        self.input_df[keys.FINAL_ADMISION_NOTE_KEY] = pd.cut(
            self.input_df[keys.FINAL_ADMISION_NOTE_KEY], note_bcket_array, include_lowest=True)

        year_bcket_array = np.array([18, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        self.input_df['edad_acceso'] = pd.cut(
            self.input_df['edad_acceso'], year_bcket_array, include_lowest=True)

        distance_bcket_array = np.array([0, 0.1, 45, 90, 135, 250, 500, 1892])
        self.input_df['distance'] = pd.cut(
            self.input_df['distance'], distance_bcket_array, include_lowest=True)

        self.final_analys_record_personal_access = self.input_df.copy()

        if self.course == 4:
            self.input_df.drop([keys.PLAN_DESCRIPTION_KEY], axis=1, inplace=True)

        self.input_df = pd.get_dummies(data=self.input_df,
                                       columns=self.input_df.drop(self.input_df.select_dtypes(
                                           include=['int64', 'float64']).columns, axis=1).columns)
        self.input_df.drop([keys.PLAN_CODE_KEY, keys.RECORD_KEY], axis=1, inplace=True)

    def analise(self):
        drop_out_data = self.input_df[self.input_df[keys.DROP_OUT_KEY] == 1]
        no_drop_out_data = self.input_df[self.input_df[keys.DROP_OUT_KEY] == 0]

        # FOR PREVIOUS AND FIRST COURSE DATA OF POLYTECHNIC
        if drop_out_data.shape[0] > no_drop_out_data.shape[0]:
            from sklearn.utils import resample
            drop_out_data_downsampled = resample(drop_out_data,
                                                 replace=False,
                                                 n_samples=no_drop_out_data.shape[0],
                                                 random_state=123)
            resample_df = pd.concat([drop_out_data_downsampled, no_drop_out_data])
            x = resample_df.drop([keys.DROP_OUT_KEY], axis=1)
            y = resample_df[keys.DROP_OUT_KEY]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25,
                                                                                    random_state=24)
        # FOR OTHER CASES.
        elif self.course > 0:
            from imblearn.combine import SMOTETomek
            x_smt, y_smt = SMOTETomek().fit_sample(self.input_df.drop([keys.DROP_OUT_KEY], axis=1),
                                                   self.input_df[keys.DROP_OUT_KEY])

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_smt, y_smt, test_size=0.25,
                                                                                    random_state=24)
        # FOR PREVIOUS AND FIRST COURSE DATA OF TEACHING
        else:
            from sklearn.utils import resample
            no_drop_out_data_downsampled = resample(no_drop_out_data,
                                                    replace=False,
                                                    n_samples=1500,
                                                    random_state=123)
            drop_out_data_upsampled = resample(drop_out_data,
                                               replace=True,
                                               n_samples=1500,
                                               random_state=123)
            resample_df = pd.concat([drop_out_data_upsampled, no_drop_out_data_downsampled])
            x = resample_df.drop([keys.DROP_OUT_KEY], axis=1)
            y = resample_df[keys.DROP_OUT_KEY]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25,
                                                                                    random_state=24)
        norm = MinMaxScaler().fit(self.x_train)
        self.x_train_norm = norm.transform(self.x_train)
        self.x_test_norm = norm.transform(self.x_test)

        best_hyperparameters_MLP = self.get_best_hyperparameters_MLP()
        self.models_developed.append(MLPClassifier(
            activation=best_hyperparameters_MLP.activation,
            solver=best_hyperparameters_MLP.solver,
            alpha=best_hyperparameters_MLP.alpha,
            learning_rate=best_hyperparameters_MLP.learning_rate, random_state=123, max_iter=1000).fit(
            self.x_train, self.y_train))

        best_hyperparameters_RForest = self.get_best_hyperparameters_RandomForest()
        self.models_developed.append(RandomForestClassifier(
            max_depth=best_hyperparameters_RForest.max_depth,
            n_estimators=best_hyperparameters_RForest.n_estimators,
            random_state=123).fit(self.x_train, self.y_train))

        best_hyperparameters_SVM = self.get_best_hyperparameters_SVM()
        self.models_developed.append(SVC(
            C=best_hyperparameters_SVM.C,
            gamma=best_hyperparameters_SVM.gamma,
            probability=True, random_state=123).fit(self.x_train_norm, self.y_train))

        for model in self.models_developed:
            if 'SVC' not in str(model):
                y_pred = model.predict(self.x_test)
                log.info(
                    "accuracy of " + str(model) + " model is: " + str(
                        sklm.accuracy_score(y_true=self.y_test, y_pred=y_pred)))
                log.info("confusion matrix of " + str(model) + "model is: \n" + str(
                    sklm.confusion_matrix(y_true=self.y_test, y_pred=y_pred)))
                log.info("recall of " + str(model) + "model is: " + str(sklm.recall_score(
                    y_true=self.y_test, y_pred=y_pred)))
            else:
                y_pred = model.predict(self.x_test_norm)
                log.info(
                    "accuracy of " + str(model) + " model is: " + str(
                        sklm.accuracy_score(y_true=self.y_test, y_pred=y_pred)))
                log.info("confusion matrix of " + str(model) + "model is: \n" + str(
                    sklm.confusion_matrix(y_true=self.y_test, y_pred=y_pred)))
                log.info("recall of " + str(model) + "model is: " + str(sklm.recall_score(
                    y_true=self.y_test, y_pred=y_pred)))

        pred_1 = self.models_developed[0].predict_proba(self.x_test)
        pred_2 = self.models_developed[1].predict_proba(self.x_test)
        pred_3 = self.models_developed[2].predict_proba(self.x_test_norm)
        weighted_ensemble_pred = pred_1 * 0.7 + pred_2 * 0.15 + pred_3 * 0.15

        y_pred = (weighted_ensemble_pred[0:, 1] >= self.threshold).astype(int)
        log.info("accuracy of ensemble model 1 is: " + str(sklm.accuracy_score(y_true=self.y_test, y_pred=y_pred)))
        log.info(
            "confusion matrix of ensemble model 1 is: \n" + str(sklm.confusion_matrix(y_true=self.y_test, y_pred=y_pred)))
        log.info("recall of ensemble model 1 is: " + str(sklm.recall_score(y_true=self.y_test, y_pred=y_pred)))

        x_test = self.input_df.drop([keys.DROP_OUT_KEY], axis=1)
        x_test_norm = norm.transform(x_test)

        pred_1 = self.models_developed[0].predict_proba(x_test)
        pred_2 = self.models_developed[1].predict_proba(x_test)
        pred_3 = self.models_developed[2].predict_proba(x_test_norm)
        weighted_ensemble_pred = pred_1 * 0.7 + pred_2 * 0.15 + pred_3 * 0.15
        y_pred = (weighted_ensemble_pred[0:, 1] >= self.threshold).astype(int)
        self.y_pred = y_pred

    def save(self):
        self.final_analys_record_personal_access[self.model_number] = self.y_pred
        self.final_analys_record_personal_access.to_csv(
            self.output_path_segment,
            sep=self.output_separator,
            index=False)


def main():
    logging.basicConfig(
        filename="debug_ensemble.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start QuadrimestersEnsemble")
    log.debug("main()")

    analys = QuadrimestersEnsemble(
        input_separator='|',
        output_separator='|'
    )
    analys.execute()


if __name__ == "__main__":
    main()
