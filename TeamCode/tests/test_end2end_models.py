


import unittest
import pytest
from ..src import sample_implementation, interface, helper_code

import os


class TestTools(unittest.TestCase):

    def setUp(self) -> None:
        self.data_folder = './TeamCode/tests/resources/example_data'
        self.model_folder = './TeamCode/tests/resources/example_model'
        self.verbose = True
        self.allowFailures = False

    def test_paths_exist(self):
        self.assertTrue(os.path.exists(self.data_folder))
        self.assertTrue(os.path.exists(self.model_folder))



    def _test_both_models(self, digitization_class, classification_class):
        self.assertTrue(issubclass(digitization_class, interface.AbstractDigitizationModel))
        self.assertTrue(issubclass(classification_class, interface.AbstractClassificationModel))

        digitization_model = digitization_class()
        classification_model = classification_class()

        ## train model
        digitization_model.train_model(self.data_folder, self.model_folder, self.verbose) 
        classification_model.train_model(self.data_folder, self.model_folder, self.verbose) 

        digitization_model = None
        classification_model = None

        ## run model
        trained_digitization_model = digitization_class.from_folder(self.model_folder, self.verbose) 
        trained_classification_model = classification_class.from_folder(self.model_folder, self.verbose) 

        records = helper_code.find_records(self.data_folder)
        num_records = len(records)

        for i in range(num_records):
            if self.verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            data_record = os.path.join(self.data_folder, records[i])


            try:
                signal = trained_digitization_model.run_digitization_model(data_record, self.verbose) 
            except:
                if self.allow_failures:
                    if self.verbose:
                        print('... digitization failed.')
                    signal = None
                else:
                    raise


            try:
                dx = trained_classification_model.run_classification_model(data_record, signal, self.verbose) 
            except:
                if self.allow_failures:
                    if self.verbose >= 2:
                        print('... dx classification failed.')
                    dx = None
                else:
                    raise

        print('finished')
        



    def test_sample_implementation(self):
        self._test_both_models(sample_implementation.ExampleDigitizationModel, sample_implementation.ExampleClassificationModel)