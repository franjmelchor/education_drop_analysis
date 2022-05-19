import argparse
from apitep_utils import FeatureSelection, ArgumentParserHelper, HypothesisTest
import logging
from apitep_utils.data_processor import DataProcessor

import keys
from data_model.school_kind import SchoolKind

log = logging.getLogger(__name__)


class QuadrimestersFeatureSelection(DataProcessor):
    quadrimester: int = None
    course: int = None
    school_kind: SchoolKind = None

    def parse_arguments(self):
        """
        Parse arguments provided via command line, and check if they are valid
        or not. Adequate defaults are provided when possible.

        Parsed arguments are:
        - paths to the input CSV datasets, separated with spaces.
        - path to the output CSV dataset.
        """

        log.info("Get integration arguments")
        log.debug("Integration.parse_arguments()")

        argument_parser = argparse.ArgumentParser(description=self.description)
        argument_parser.add_argument("-i", "--input_path", required=True,
                                     help="path to the input CSV dataset")
        argument_parser.add_argument("-o", "--output_path", required=True,
                                     help="path to the output CSV dataset")
        argument_parser.add_argument("-s", "--school_kind", required=True,
                                     help="school kind to analyze")
        argument_parser.add_argument("-c", "--course", required=True,
                                     help="course to analyze")
        argument_parser.add_argument("-q", "--quadrimester", required=True,
                                     help="quadrimester of course to analyze")

        arguments = argument_parser.parse_args()
        self.input_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.input_path)
        self.output_path_segment = ArgumentParserHelper.parse_data_file_path(
            data_file_path=arguments.output_path,
            check_is_file=False)
        self.course = int(arguments.course)
        self.quadrimester = int(arguments.quadrimester)
        school_kind_str = arguments.school_kind
        if school_kind_str == "Teaching":
            self.school_kind = SchoolKind.Teaching
        elif school_kind_str == "Polytechnic":
            self.school_kind = SchoolKind.Polytechnic

    def process(self):
        """
        Feature selection of analys_record_personal_access
        """
        if self.course == 0:
            log.info("Feature Selection of analys_record_personal_access data of school: " + self.school_kind.value)
        else:
            log.info("Feature Selection of analys_record_personal_access data of school: " + self.school_kind.value +
                     " with data of course " + str(self.course) + " and quadrimester " + str(self.quadrimester))

        log.debug("QuadrimestersFeatureSelection.process()")

        dependency_tests = []

        dv_analys_record_personal_access = self.input_df.drop([keys.RECORD_KEY, keys.PLAN_CODE_KEY], axis=1)

        if self.course == 0:
            categorical_cols = dv_analys_record_personal_access.drop([keys.FINAL_ADMISION_NOTE_KEY], axis=1).columns
        elif self.course == 1:
            categorical_cols = dv_analys_record_personal_access.drop(
                [keys.FINAL_ADMISION_NOTE_KEY, keys.CUM_PASS_RATIO_KEY, keys.CUM_ABSENT_RATIO_KEY, keys.CUM_MEDIAN_KEY],
                axis=1).columns
        else:
            categorical_cols = dv_analys_record_personal_access.drop(
                [keys.FINAL_ADMISION_NOTE_KEY, keys.CUM_PASS_RATIO_KEY, keys.CUM_ABSENT_RATIO_KEY, keys.CUM_MEDIAN_KEY,
                 keys.CUM_MORE_1ST_CALL_RATIO_KEY],
                axis=1).columns

        for col in categorical_cols:
            chi2_test = HypothesisTest(
                dataframe=dv_analys_record_personal_access,
                test_type=HypothesisTest.TestType.Chi2,
                target=dv_analys_record_personal_access[keys.DROP_OUT_KEY],
                candidates=[dv_analys_record_personal_access[col]],
                significance_value=0.1
            )
            dependency_tests.append(chi2_test)

        numerical_cols = list(set(dv_analys_record_personal_access.columns) - set(categorical_cols))

        for col in numerical_cols:
            spearman_test = HypothesisTest(
                dataframe=dv_analys_record_personal_access,
                test_type=HypothesisTest.TestType.Spearman,
                target=dv_analys_record_personal_access[keys.DROP_OUT_KEY],
                candidates=[dv_analys_record_personal_access[col]]
            )
            dependency_tests.append(spearman_test)

        feature_selection = FeatureSelection(
            dependency_tests=dependency_tests
        )
        feature_selection.process()

        final_colums = feature_selection.influencing_features
        final_colums.append(keys.RECORD_KEY)
        final_colums.append(keys.PLAN_CODE_KEY)
        if keys.PLAN_DESCRIPTION_KEY not in final_colums:
            final_colums.append(keys.PLAN_DESCRIPTION_KEY)

        self.output_df = self.input_df[final_colums]


def main():
    logging.basicConfig(
        filename="debug.log",
        level=logging.DEBUG,
        format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    log.info("--------------------------------------------------------------------------------------")
    log.info("Start QuadrimestersFeatureSelection")
    log.debug("main()")

    feat_sel = QuadrimestersFeatureSelection(
        input_separator='|',
        output_separator='|'
    )
    feat_sel.parse_arguments()
    feat_sel.load()
    feat_sel.process()
    feat_sel.save()


if __name__ == "__main__":
    main()
