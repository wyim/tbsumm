import unittest
import sys

sys.path.append('..')
from tb_records import update_event, start_event, TbGradedRubric
import pandas as pd


class TestUpdateEvent(unittest.TestCase):

    def test_rubricevent_singleton(self):
        event = {}
        update_event( event, [], False)

        event_expected = {
            'attribute_classifications':[],
            'attribute_importances':[]
        }
        self.assertEqual( len(event.keys()), 2 )
        self.assertDictEqual( event, event_expected )

    def test_rubricevent_1mention_nonaddedfact(self):

        item1 = {
            'Unnamed: 2': 'stage IA', #event_argument
            'Unnamed: 3': 'mention', #fact_category
            'Unnamed: 4': 'critical', #fact_importance
            'Unnamed: 6': 'present', #fact_presence
            'Unnamed: 7': pd.NA, #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        event = {}
        update_event( event, [pd.Series(item1)] , False)

        self.assertEqual( event['mention'], item1['Unnamed: 2'] )
        self.assertEqual( event['mention_classification'], 'ENTAILED_BY' )
        self.assertEqual( event['mention_importance'], 'critical' )

        self.assertEqual( len(event['attributes']), 0 )
        self.assertEqual( len(event['attribute_classifications']), 0 )
        self.assertEqual( len(event['attribute_importances']), 0 )
    

        item1 = {
            'Unnamed: 2': 'stage IA', #event_argument
            'Unnamed: 3': 'mention', #fact_category
            'Unnamed: 4': 'critical', #fact_importance
            'Unnamed: 6': 'not_present', #fact_presence
            'Unnamed: 7': pd.NA, #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        event = {}
        update_event( event, [pd.Series(item1)] , False)

        self.assertEqual( event['mention'], item1['Unnamed: 2'] )
        self.assertEqual( event['mention_classification'], 'OTHER_NOT_SUPPORTED' )
        self.assertEqual( event['mention_importance'], 'critical' )

        self.assertEqual( len(event['attributes']), 0 )
        self.assertEqual( len(event['attribute_classifications']), 0 )
        self.assertEqual( len(event['attribute_importances']), 0 )
    
    def test_rubricevent_1mention_addedfact(self):

        item1 = {
            'Unnamed: 2': 'stage IA', #event_argument
            'Unnamed: 3': pd.NA, #fact_category
            'Unnamed: 4': pd.NA, #fact_importance
            'Unnamed: 6': pd.NA, #fact_presence
            'Unnamed: 7': 'accurate', #fact_accuracy
            'Unnamed: 8': 'critical', #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        event = {}
        update_event( event, [pd.Series(item1)] , True)

        self.assertEqual( event['mention'], None )
        self.assertEqual( event['mention_classification'], None )
        self.assertEqual( event['mention_importance'], None )

        self.assertEqual( len(event['attributes']), 1 )
        self.assertEqual( event['attribute_classifications'], ['ENTAILED_BY'] )
        self.assertEqual( event['attribute_importances'], ['critical'] )
    
        item1 = {
            'Unnamed: 2': 'stage IA', #event_argument
            'Unnamed: 3': pd.NA, #fact_category
            'Unnamed: 4': pd.NA, #fact_importance
            'Unnamed: 6': pd.NA, #fact_presence
            'Unnamed: 7': 'not_accurate', #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': 'minimal' #hallucination_harmfullness
        }
        event = {}
        update_event( event, [pd.Series(item1)] , True)

        self.assertEqual( event['mention'], None )
        self.assertEqual( event['mention_classification'], None )
        self.assertEqual( event['mention_importance'], None )

        self.assertEqual( len(event['attributes']), 1 )
        self.assertListEqual( event['attribute_classifications'], ['OTHER_NOT_SUPPORTED'] )
        self.assertListEqual( event['attribute_importances'], ['minimal'] )

    def test_rubricevent_event_nonaddedfact(self):

        item1 = {
            'Unnamed: 2': 'resection', #event_argument
            'Unnamed: 3': 'mention', #fact_category
            'Unnamed: 4': 'critical', #fact_importance
            'Unnamed: 6': 'present', #fact_presence
            'Unnamed: 7': pd.NA, #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        item2 = {
            'Unnamed: 2': 'in 2009', #event_argument
            'Unnamed: 3': 'attribute', #fact_category
            'Unnamed: 4': 'noncritical', #fact_importance
            'Unnamed: 6': 'present', #fact_presence
            'Unnamed: 7': pd.NA, #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        item3 = {
            'Unnamed: 2': 'left lower lobe', #event_argument
            'Unnamed: 3': 'attribute', #fact_category
            'Unnamed: 4': 'critical', #fact_importance
            'Unnamed: 6': 'present', #fact_presence
            'Unnamed: 7': pd.NA, #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        event = {}
        row_cache = [ pd.Series(item1), pd.Series(item2), pd.Series(item3) ]
        update_event( event, row_cache , False)

        self.assertEqual( event['mention'], item1['Unnamed: 2'] )
        self.assertEqual( event['mention_classification'], 'ENTAILED_BY' )
        self.assertEqual( event['mention_importance'], 'critical' )

        self.assertEqual( len(event['attributes']), 2 )
        self.assertEqual( len(event['attribute_classifications']), 2 )
        self.assertEqual( len(event['attribute_importances']), 2 )

        self.assertListEqual( event['attributes'], [item2['Unnamed: 2'],item3['Unnamed: 2']])
        self.assertListEqual( event['attribute_classifications'], ['ENTAILED_BY','ENTAILED_BY'])
        self.assertListEqual( event['attribute_importances'], [item2['Unnamed: 4'],item3['Unnamed: 4']])

    def test_rubricevent_event_addedfact(self):

        item2 = {
            'Unnamed: 2': 'in 2009', #event_argument
            'Unnamed: 3': pd.NA, #fact_category
            'Unnamed: 4': pd.NA, #fact_importance
            'Unnamed: 6': pd.NA, #fact_presence
            'Unnamed: 7': 'accurate', #fact_accuracy
            'Unnamed: 8': 'noncritical', #addedfact_importance
            'Unnamed: 9': pd.NA #hallucination_harmfullness
        }
        item3 = {
            'Unnamed: 2': 'left lower lobe', #event_argument
            'Unnamed: 3': pd.NA, #fact_category
            'Unnamed: 4': pd.NA, #fact_importance
            'Unnamed: 6': pd.NA, #fact_presence
            'Unnamed: 7': 'not_accurate', #fact_accuracy
            'Unnamed: 8': pd.NA, #addedfact_importance
            'Unnamed: 9': 'medium' #hallucination_harmfullness
        }
        event = {}
        row_cache = [ pd.Series(item2), pd.Series(item3) ]
        update_event( event, row_cache , True)

        self.assertEqual( event['mention'], None )
        self.assertEqual( event['mention_classification'], None )
        self.assertEqual( event['mention_importance'], None )

        self.assertEqual( len(event['attributes']), 2 )
        self.assertEqual( len(event['attribute_classifications']), 2 )
        self.assertEqual( len(event['attribute_importances']), 2 )

        self.assertListEqual( event['attributes'], [item2['Unnamed: 2'],item3['Unnamed: 2']])
        self.assertListEqual( event['attribute_classifications'], ['ENTAILED_BY','OTHER_NOT_SUPPORTED'])
        self.assertListEqual( event['attribute_importances'], [item2['Unnamed: 8'],item3['Unnamed: 9']])

class TestStartEvent(unittest.TestCase):

    def test_rubricevent_singleton_nonadded(self):
       
        is_added_fact = False
        event_text = '73yo'
        fact_category='demographics'
        fact_importance='noncritical'
        fact_presence=pd.NA
        fact_accuracy=pd.NA
        addedfact_importance=pd.NA
        hallucination_harmfullness=pd.NA
        event = start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness)
        
        event_expected = {'text':event_text, 'type':fact_category, 'attributes':[] }

        is_added_fact = False
        event_text = '73yo'
        fact_category='demographics'
        fact_importance='critical'
        fact_presence='present'
        fact_accuracy=pd.NA
        addedfact_importance=pd.NA
        hallucination_harmfullness=pd.NA
        event = start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness)
        
        event_expected = {'text':event_text,'type':fact_category,'attributes':[],
                'mention':event_text, 
                'mention_importance':fact_importance,
                'mention_classification': 'ENTAILED_BY' }
        
        self.assertDictEqual( event, event_expected )

        is_added_fact = False
        event_text = '73yo'
        fact_category='demographics'
        fact_importance='noncritical'
        fact_presence='not_present'
        fact_accuracy=pd.NA
        addedfact_importance=pd.NA
        hallucination_harmfullness=pd.NA
        event = start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness)
        
        event_expected = {'text':event_text,'type':fact_category,'attributes':[],
                'mention':event_text, 
                'mention_importance':fact_importance,
                'mention_classification': 'OTHER_NOT_SUPPORTED' }
        
        self.assertDictEqual( event, event_expected )
    

    def test_rubricevent_singleton_added(self):
        is_added_fact = True
        event_text = '73yo'
        fact_category=pd.NA
        fact_importance=pd.NA
        fact_presence=pd.NA
        fact_accuracy='accurate'
        addedfact_importance='critical'
        hallucination_harmfullness=pd.NA
        event = start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness)
        
        event_expected = {'text':event_text,'type':fact_category,'attributes':[],
                        'mention':event_text,
                        'mention_importance':addedfact_importance,
                        'mention_classification': 'ENTAILED_BY' }
        
        self.assertDictEqual( event, event_expected )

        is_added_fact = True
        event_text = '73yo'
        fact_category=pd.NA
        fact_importance=pd.NA
        fact_presence=pd.NA
        fact_accuracy='not_accurate'
        addedfact_importance=pd.NA
        hallucination_harmfullness='severe'

        event = start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness)
        
        event_expected = {'text':event_text,'type':fact_category,'attributes':[],
                        'mention':event_text,
                        'mention_importance': hallucination_harmfullness,
                        'mention_classification': 'OTHER_NOT_SUPPORTED' }
        
        self.assertDictEqual( event, event_expected )
        is_added_fact = True
        event_text = '73yo'
        fact_category=pd.NA
        fact_importance=pd.NA
        fact_presence=pd.NA
        fact_accuracy=pd.NA
        addedfact_importance=pd.NA
        hallucination_harmfullness=pd.NA
        event = start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness)
        
        event_expected = {'text':event_text,'type':fact_category,'attributes':[],
                        'mention':event_text,
                        'mention_importance': hallucination_harmfullness,
                        'mention_classification': 'OTHER_NOT_SUPPORTED' }
        
        self.assertDictEqual( event, event_expected )


class TestTbRecordScoreDirectional(unittest.TestCase):

    def test_scoring_simple(self):
        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)
    
    def test_scoring_masks(self):
        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1*0+1+0.5)/3
        additional_weights = [0,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+0+0.5+0)/3
        additional_weights = [1,1,1,0]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+0+0.5*0.5+0)/3
        additional_weights = [1,1,0.5,0]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)


    def test_scoring_adderror(self):
        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1-0.25+.5)/4
        additional_weights = [1,1,1,1]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1-0.25+.5+.5)/4
        additional_weights = [1,1,1,1]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)
    
    def test_scoring_complex(self):
        facts = [
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0.5)/4
        additional_weights = [1,1,1,1,0]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (-.25*1+1*.2+0.5+0+0)/4
        additional_weights = [1,.2,1,1,0]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)


        facts = [
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['CONTRADICTED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (-.25*1 + 1*.2 + 0.5 + 0 + 0 + .5*.3)/5
        additional_weights = [1,.2,1,1, 0,.3]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

    def test_scoring_negative(self):
        facts = [
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':[] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'CONTRADICTED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [1,1,1,1,1]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = 0
        additional_weights = [1,1,1,1,0]
        error_weight=-0.25
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)
    
    def test_scoring_nonoriginal(self):
        facts = [
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':[] },
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'OTHER_NOT_SUPPORTED', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [0,0,0,0]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': None, 'attribute_classifications':[] },
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [0,0,0,0]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0.5+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)


        facts = [
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': None, 'attribute_classifications':[] },
          {'mention_classification': None, 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
          {'mention_classification': 'ENTAILED_BY', 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [0,0,0,0]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)

        facts = [
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY'] },
          {'mention_classification': None, 'attribute_classifications':[] },
          {'mention_classification': None, 'attribute_classifications':['OTHER_NOT_SUPPORTED'] },
          {'mention_classification': None, 'attribute_classifications':['ENTAILED_BY','OTHER_NOT_SUPPORTED'] },
        ]
        is_original_rubric = [0,0,0,0]
        tbgradedrubric = TbGradedRubric('HYP','REF',facts,is_original_rubric)

        score_expected = (1+1+0+0.5)/4
        additional_weights = [1,1,1,1]
        error_weight=0
        score = tbgradedrubric.score_directional(additional_weights=additional_weights,error_weight=error_weight)

        self.assertEqual(score_expected,score)
    

if __name__ == '__main__':
    unittest.main()