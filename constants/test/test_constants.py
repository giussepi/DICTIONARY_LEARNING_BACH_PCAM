# -*- coding: utf-8 -*-
""" test.test_constants """

import unittest

from constants.constants import Label
from core.exceptions.dataset import LabelIdInvalid


class LabelTestCase(unittest.TestCase):

    def test_is_valid_option(self):
        self.assertTrue(Label.is_valid_option(0))
        self.assertFalse(Label.is_valid_option(4))

    def test_get_name(self):
        self.assertRaises(LabelIdInvalid, Label.get_name, 4)
        self.assertEqual(Label.get_name(0), Label.NORMAL.name)

    def test_get_choices_as_string(self):
        self.assertEqual(
            Label.get_choices_as_string(),
            '0 : Normal, 1 : Benign, 2 : In Situ, 3 : Invasive'
        )
