import unittest
import sys

sys.path.append('..')
from tb_eval import fix_entailmentclassifications, harmonize_classification, harmonize_entailmentclassifications, convert_to_tbgradedrubric
from tb_eval import get_rubric_score, get_rubric_score_precision, get_tbfact_score
import pandas as pd


class TestHarmonizeFunctions(unittest.TestCase):

    def test_harmonize_classification(self):
        classification1 = 'ENTAILED_BY'
        classificaiton2 = 'ENTAILED_BY'
        harmonized = harmonize_classification( classification1, classificaiton2 )
        expected_classification = 'ENTAILED_BY'
        self.assertEqual( harmonized, expected_classification )

        classification1 = 'CONTRADICTED_BY'
        classificaiton2 = 'ENTAILED_BY'
        harmonized = harmonize_classification( classification1, classificaiton2 )
        expected_classification = 'ENTAILED_BY'
        self.assertEqual( harmonized, expected_classification )

        classification1 = 'ENTAILED_BY'
        classificaiton2 = 'CONTRADICTED_BY'
        harmonized = harmonize_classification( classification1, classificaiton2 )
        expected_classification = 'ENTAILED_BY'
        self.assertEqual( harmonized, expected_classification )

        classification1 = 'OTHER_NOT_SUPPORTED'
        classificaiton2 = 'CONTRADICTED_BY'
        harmonized = harmonize_classification( classification1, classificaiton2 )
        expected_classification = 'OTHER_NOT_SUPPORTED'
        self.assertEqual( harmonized, expected_classification )

        classification1 = 'OTHER_NOT_SUPPORTED'
        classificaiton2 = 'RELEVANT_BUT_NOT_SUPPORTED'
        harmonized = harmonize_classification( classification1, classificaiton2 )
        expected_classification = 'OTHER_NOT_SUPPORTED'
        self.assertEqual( harmonized, expected_classification )
    

    def test_harmonize_entailmentclassification(self):
        entailmentclassification1 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['RELEVANT_BUT_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        entailmentclassification2 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','ENTAILED_BY'] },
        ]

        harmonize_entailmentclassifications(entailmentclassification1,entailmentclassification2)

        expected_entailmentclassifications = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['RELEVANT_BUT_NOT_SUPPORTED','ENTAILED_BY'] },
        ]

        self.assertListEqual( entailmentclassification1, expected_entailmentclassifications )
    
    def test_fix_entailmentclassifications(self):
        entailmentclassification1 = [
            {'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY' },
            {'mention_classification': 'blah', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['blah','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        fix_entailmentclassifications( entailmentclassification1 )

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]

        self.assertListEqual( entailmentclassification1, expected_entailmentclassifications )


class TestRubricScoring(unittest.TestCase):

    def test_rubricscoring(self):

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        error_weight = 0.0
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = 1/4
        self.assertEqual( score, expected_score )

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        error_weight = 0.0
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = (0 + 1 + .5)/4
        self.assertEqual( score, expected_score )

    def test_rubricscoring_negatives(self):
        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        error_weight = -.5
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = 1/4
        self.assertEqual( score, expected_score )

        expected_entailmentclassifications = [
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        error_weight = -.5
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = 0
        self.assertEqual( score, expected_score )

    def test_rubricscoring_alterweightlist(self):
        expected_entailmentclassifications = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True, False, True,True]
        error_weight = 0
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = (1+0+0+.5)/3
        self.assertEqual( score, expected_score )

        expected_entailmentclassifications = [
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True, True, False, True,True]
        error_weight = -.3
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = (-.3+1+0+0+.5)/4
        self.assertEqual( score, expected_score )

        expected_entailmentclassifications = [
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True, True, False, True,True]
        error_weight = -.3
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = (-.3+1+0+0+.5)/4
        self.assertEqual( score, expected_score )

        
        expected_entailmentclassifications = [
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True, True, False, True,True]
        error_weight = -.3
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = (-.3+ .5 + 0 + 0 + .5)/4
        self.assertEqual( score, expected_score )


        expected_entailmentclassifications = [
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True, True, False, True,False]
        error_weight = -.3
        score = get_rubric_score( tbgradedrubric, error_weight )

        expected_score = (-.3+ .5 + 0 + 0 + 0)/3
        self.assertEqual( score, expected_score )


class TestRubricScoringPrecision(unittest.TestCase):

    def test_rubricscoringprecision_noerror(self):

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True,True,False,False]

        error_weight = 0.0
        score = get_rubric_score_precision( tbgradedrubric, error_weight )

        expected_score = 1/3
        self.assertEqual( score, expected_score )


        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True,True,False,False,False]

        error_weight = 0.0
        score = get_rubric_score_precision( tbgradedrubric, error_weight )

        expected_score = (0 + .5 + 0 + 1.0 ) /4
        self.assertEqual( score, expected_score )

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True,True,False,False,False]

        error_weight = 0.0
        score = get_rubric_score_precision( tbgradedrubric, error_weight )

        expected_score = (0 + .5 + 0 + 1.0 ) /4
        self.assertEqual( score, expected_score )
    
    def test_rubricscoringprecision_error(self):

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True,True,False,False]

        error_weight = -.2
        score = get_rubric_score_precision( tbgradedrubric, error_weight )

        expected_score = 1/3
        self.assertEqual( score, expected_score )


        expected_entailmentclassifications = [
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True,True,False,False,False]

        error_weight = -.2
        score = get_rubric_score_precision( tbgradedrubric, error_weight )

        expected_score = (0 + .5 + 0 + 1.0 ) /4
        self.assertEqual( score, expected_score )

        expected_entailmentclassifications = [
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric = convert_to_tbgradedrubric(expected_entailmentclassifications)
        tbgradedrubric.is_original_rubric = [True,True,False,False,False]

        error_weight = -.2
        score = get_rubric_score_precision( tbgradedrubric, error_weight )

        expected_score = (0 + .5 + 0 + 1.0 -.2 ) /4
        self.assertEqual( score, expected_score )


class TestTbFactScoring(unittest.TestCase):

    def test_tbfact_noerror(self):

        expected_entailmentclassifications1 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric1 = convert_to_tbgradedrubric(expected_entailmentclassifications1)

        expected_entailmentclassifications2 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric2 = convert_to_tbgradedrubric(expected_entailmentclassifications2)

        error_weight = 0.0
        precision = (.5+ 0 + 1+.5+0)/5
        recall = (1+.5+0+1+.5+0)/6
        expected_score = 2*precision*recall/(precision+recall)

        score = get_tbfact_score( tbgradedrubric1, tbgradedrubric2, error_weight )

        self.assertEqual( score, expected_score )
    
    def test_tbfact_error(self):

        expected_entailmentclassifications1 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric1 = convert_to_tbgradedrubric(expected_entailmentclassifications1)

        expected_entailmentclassifications2 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric2 = convert_to_tbgradedrubric(expected_entailmentclassifications2)

        error_weight = -.25
        precision = (.5 -.25 + 1+.5+0)/5
        recall = (1+.5+ -.25 +1+.5+0)/6
        expected_score = 2*precision*recall/(precision+recall)

        score = get_tbfact_score( tbgradedrubric1, tbgradedrubric2, error_weight )

        self.assertEqual( score, expected_score )

    def test_tbfact_missingmentions(self):

        expected_entailmentclassifications1 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': None, 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY','ENTAILED_BY'] },
        ]
        tbgradedrubric1 = convert_to_tbgradedrubric(expected_entailmentclassifications1)
        tbgradedrubric1.is_original_rubric = [1,1,1,0,0]

        expected_entailmentclassifications2 = [
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
            {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
            {'mention_classification': None, 'attribute_classifications':['OTHER_NOT_SUPPORTED','RELEVANT_BUT_NOT_SUPPORTED'] },
        ]
        tbgradedrubric2 = convert_to_tbgradedrubric(expected_entailmentclassifications2)
        tbgradedrubric2.is_original_rubric = [1,1,1,1,1,0]

        error_weight = 0.0
        precision = (.5 + 0 + 1+ 0 + 1 )/5
        recall = ( 1 + .5 + 0 + 1 + .5 + 0 )/6
        expected_score = 2*precision*recall/(precision+recall)

        score = get_tbfact_score( tbgradedrubric1, tbgradedrubric2, error_weight )

        self.assertEqual( score, expected_score )

if __name__ == '__main__':
    unittest.main()