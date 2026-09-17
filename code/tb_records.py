import sys
import pandas as pd


FACT_CATEGORY = [
    'demographic',
    'family_history',
    'stage',
    'diagnosis',
    'performance_status',
    'symptom',
    'treatment_systemic',
    'treatment_surgery',
    'treatment_radiation',
    'treatment_other',
    'test_imaging',
    'testresult_labs',
    'testresult_molecular',
    'testresult_pathology',
    'open_question',
    'other',
]

FACT_IMPORTANCE = ['critical','noncritical']
FACT_PRESENCE = [
    'present',
    'partially_present',
    'not_present'
]
FACT_ACCURACY =[
    'accurate',
    'not_accurate'
]
ADDEDFACT_CRITICALITY = ['critical','noncritical']
HALLUCINATION_HARMFULNESS = ['severe','medium','minimal']

SECTIONS = ['Summary','Oncological History','Considered Options','Supporting data']
RATING_METRICS = ['completeness','factual-accuracy','relevance','overall']

ENTAILMENT_CLASSIFICATIONS = ['ENTAILED_BY','CONTRADICTED_BY','RELEVANT_BUT_NOT_SUPPORTED','OTHER_NOT_SUPPORTED']

SCORING_SCHEME = {
    'ENTAILED_BY':1.0,
    'CONTRADICTED_BY':-0.25,
    'RELEVANT_BUT_NOT_SUPPORTED':0.0,
    'OTHER_NOT_SUPPORTED':0.0
}


class TbSummary:
    """
    Class to include TB summary.
    """
    def __init__(self,data_dict,start_index=0):

        self.case_id = data_dict['case_id']
        self.author_id = data_dict['author_id']

        self.summary = data_dict['summary'][start_index]
        self.summary_lib = ['text','reason','comments']

        self.oncological_history = data_dict['oncological_history'][start_index:]
        self.oncological_history_lib = ['date','description','comments']

        self.considered_options = data_dict['considered_options'][start_index:]
        self.considered_options_lib = ['from_initial',
                                       'description_shortname',
                                       'description',
                                       'links',
                                       'pursing','rationale','comments']

        self.supporting_data = data_dict['supporting_data'][start_index:]
        self.supporting_data_lib = ['docid','text','type','comments']

    def get_attribute(self,attribute_category,attribute) :
        """
        Use this only if we want to get specific column of a particular overall category.
        This is not strongly checked.

        Returns a list.
        """
        lst = getattr(self, attribute_category)
        attribute_category_lib = getattr(self, attribute_category+'_lib')
        index = attribute_category_lib.index(attribute)
        return [ x[index] for x in lst ]
    
    def to_str(self,attribute_category):
        if attribute_category=='summary':
            return self.summary[0]
        
        if attribute_category=='oncological_history':
            lines = []
            for row in self.oncological_history :
                date = row[0]
                description = row[1]
                lines.append( '{}: {}'.format(date,description) )
            return '\n\n'.join(lines)
        
        if attribute_category=='considered_options':
            lines = []
            for row in self.considered_options :
                description_shortname = row[1]
                description = row[2]
                links = row[3]
                lines.append( '{} - {}\n({})'.format( description_shortname,
                                                      description,
                                                      links) )
            return '\n\n'.join(lines)
        
        if attribute_category=='supporting_data':
            lines = []
            for row in self.supporting_data :
                docid = row[0]
                text = row[1]
                type = row[1]
                lines.append( '{}: {}({})'.format(docid,text,type) )
            return '\n\n'.join(lines)
    
    def to_json(self) :
        return {
            'case_id': self.case_id,
            'author_id': self.author_id,
            'summary': self.summary,
            'oncological_history': self.oncological_history,
            'considered_options': self.considered_options,
            'supporting_data': self.supporting_data,
            'library': {
                'summary': self.summary_lib,
                'oncological_history': self.oncological_history_lib,
                'considered_options': self.considered_options_lib,
                'supporting_data': self.supporting_data_lib
            }
        }

    @classmethod
    def read_from_json(self,json_obj) :
        data_dict = {
            'case_id': json_obj['case_id'],
            'author_id': json_obj['author_id'], 
            'summary': [json_obj['summary']],
            'supporting_data': json_obj['supporting_data'],
            'considered_options': json_obj['considered_options'],
            'oncological_history': json_obj['oncological_history']
        }
        return TbSummary(data_dict,start_index=0)
    
    @classmethod
    def read_from_sheet(cls, case_id, author_id, df_sheet):

        summary = []
        supporting_data = []
        considered_options = []
        oncological_history = []

        start_supportdata = df_sheet[ df_sheet['Summary']=='Supporting Data'].index[0]
        start_consideredoptions = df_sheet[ df_sheet['Summary']=='Considered Options'].index[0]
        start_oncologicalhx = df_sheet[ df_sheet['Summary']=='Oncological History'].index[0]

        for index, row in df_sheet.iterrows() :

            if 0<=index and index<start_supportdata :
                if pd.isnull( row['Unnamed: 1'] ) :
                    continue
                text = row['Unnamed: 1']
                reason = row['Unnamed: 2']
                comments = row['Unnamed: 3']
                summary.append( [text,reason,comments] )
            
            if start_supportdata<index and index<start_consideredoptions :
                if pd.isnull( row['Unnamed: 1'] ) :
                    continue
                docid = row['Summary']
                text = row['Unnamed: 1']
                type = row['Unnamed: 2']
                comments = row['Unnamed: 3']
                supporting_data.append( [docid,text,type,comments] )

            if start_consideredoptions<index and index<start_oncologicalhx :
                if pd.isnull( row['Unnamed: 1'] ) :
                    continue
                from_initial = row['Summary']
                description_shortname = row['Unnamed: 1']
                description = row['Unnamed: 2']
                links = row['Unnamed: 3']
                pursing = row['Unnamed: 4']
                rationale = row['Unnamed: 5'] if 'Unnamed: 5' in row else ''
                comments = row['Unnamed: 6'] if 'Unnamed: 6' in row  else ''
                considered_options.append( [from_initial,description_shortname,description,links,pursing,rationale,comments] )
            
            if start_oncologicalhx<index :
                if pd.isnull( row['Unnamed: 1'] ) :
                    continue
                date = row['Summary']
                description = row['Unnamed: 1']
                comments = row['Unnamed: 2']
                oncological_history.append( [date,description,comments] )
        
        data_dict = {
            'case_id': case_id,
            'author_id': author_id, 
            'summary': summary,
            'supporting_data': supporting_data,
            'considered_options': considered_options,
            'oncological_history': oncological_history
        }
        return TbSummary(data_dict,start_index=1)
    

class TbGradedRubric:
    """
    Class to include TB rubric and its CANDIDATE/REFERENCE text.
    This is used for fine-grained grading of free text.

    Includes 4 esential objects:
    1. candidate_text: free text to be graded
    2. reference_text: free text to be used as ground truth
    3. facts: list of facts where mention is entailed
        { "text":"PSA undected, posoperatively",
          "type":"biomarkers",
          "mention":"PSA", "attributes":["undetectable","postoperatively"],
          "mention_classification":"ENTAILED_BY", "mention_classification_reason": "PSA after surgery undetectable was mentioned exactly",
          "attribute_classifications":["ENTAILED_BY","ENTAILED_BY"], "attribute_classifications_reasons":["PSA after surgery undetectable was mentioned exactly","PSA after surgery undetectable was mentioned exactly"] },
    7. is_original_rubric: list of True/False parallel to facts to mark if fact was from a rubric (only useful for hand-crafted facts)
    """
    def __init__(self, candidate_text, reference_text, 
                       facts,
                       is_original_rubric=None
                       ):
        self.candidate_text = candidate_text
        self.reference_text = reference_text
        self.facts = facts
        self.is_original_rubric = is_original_rubric
    
    def score_directional(self, additional_weights=None, error_weight=-0.25) :
        """
        Grade-school scoring v1.0
        RECALL based ( 1.0, 0.5, 0.0 ) at the fact level.

        For gold, use is_original_rubric to screen for the original rubric criteria.
        To add punishment, set error_weight to less than 1.

        Without punishment:
        1.0 is assigned if mention and all attributes are entailed
        0.5 partial entailment is assigned if the mention is entailed, and some attributes are not
        0.0 otherwise

        With punishment:
        Adds user-adjusted error_weight for each mention that is contradicted.
        """
        score = 0.0
        additional_weights = additional_weights if additional_weights else [1]*len(self.facts)
        non_zero = 0

        for ind, fact in enumerate(self.facts) :
            addw = additional_weights[ind]
            if addw==0:
                continue
            non_zero+=1
            #deductions from the unsupported facts
            if fact['mention_classification']=='CONTRADICTED_BY':
                score += error_weight*addw
            elif (fact['mention_classification']=='ENTAILED_BY'):
                attribute_classifications = set( fact['attribute_classifications'] )
                fact_weight = 1.0 if ( len(attribute_classifications)==0 ) or (  ( len(attribute_classifications)==1 ) and ( fact['attribute_classifications'][0]== 'ENTAILED_BY') ) else 0.5
                score += fact_weight*addw
            elif (not self.is_original_rubric[ind]) and pd.isna( fact['mention_classification']):
                attribute_classifications = set( fact['attribute_classifications'] )
                if ( len(attribute_classifications)==0 ) or (  ( len(attribute_classifications)==1 ) and ( fact['attribute_classifications'][0]== 'ENTAILED_BY') ) :
                    fact_weight = 1.0
                elif ( len(attribute_classifications)>1 ) and ( 'ENTAILED_BY' in attribute_classifications ):
                    fact_weight = 0.5
                else:
                    fact_weight = 0.0
                score += fact_weight*addw
        if len(self.facts)==0:
            return 1.0
        return max(score/non_zero,0)

    def to_json(self) :
        return {
            'candidate_text': self.candidate_text,
            'reference_text': self.reference_text,
            'facts': self.facts,
            'is_original_rubric': self.is_original_rubric
        }
    
    @classmethod
    def read_from_json(self,json_obj) :
        return TbGradedRubric( json_obj['candidate_text'],
                               json_obj['reference_text'],
                               json_obj['facts'],
                               json_obj['is_original_rubric'])

    @classmethod
    def read_from_sheet(cls, candidate_text, reference_text, df_sheet):

        facts = []
        is_original_rubricitem = []
        event = None

        row_cache = []

        start_rubric = df_sheet[ df_sheet['RUBRIC CREATION']=='rubric_criterion' ].index[0]
        start_addedinfo = df_sheet[ df_sheet['RUBRIC CREATION']=='[ADDED INFORMATION  - in rubric grading]' ].index[0]

        for index, row in df_sheet.iterrows() :

            if index<=start_rubric :
                continue

            #dependending on sheet location, then save the is_added_fact
            is_added_fact = False if index<start_addedinfo else True

            event_text = row['Unnamed: 1']
            event_argument = row['Unnamed: 2']
            fact_category = str(row['Unnamed: 3']).split(' ')[0]
            fact_importance = row['Unnamed: 4']
            fact_presence = row['Unnamed: 6']
            fact_accuracy = row['Unnamed: 7']
            addedfact_importance = row['Unnamed: 8']
            hallucination_harmfullness = row['Unnamed: 9']

            if not pd.isna( event_text ) :
                #check if event has started (if not, which happens at the beginning to dont add event)
                #add event when new event starts
                if event :
                    update_event( event, row_cache, is_added_fact )
                    is_original_rubricitem.append(not is_added_fact)
                    facts.append( event )

                event = start_event( is_added_fact, event_text, fact_category, fact_importance, fact_presence, fact_accuracy, addedfact_importance, hallucination_harmfullness)
                row_cache = []
            
            # if not the header for added information or blank information
            elif (index!=start_addedinfo) and ( (not pd.isna(event_text)) or (not pd.isna(event_argument)) ):
                row_cache.append(row)
        
        #add event at tail
        update_event( event, row_cache, is_added_fact )
        is_original_rubricitem.append(not is_added_fact)
        facts.append( event )

        return TbGradedRubric(candidate_text,reference_text,facts,is_original_rubricitem)


def start_event( is_added_fact,
                event_text,
                fact_category, fact_importance,
                fact_presence,
                fact_accuracy, addedfact_importance, hallucination_harmfullness) :
    #mentions will be written over if there is a later explicit mention
    if not is_added_fact :
        #no fact_presense should be case where there is an event that splits into mentions
        if pd.isna( fact_presence ) :
            event = {'text':event_text, 'type':fact_category, 'attributes':[] }
        else :
            #this case of singleton event
            mention_classification = 'ENTAILED_BY' if fact_presence=='present' else 'OTHER_NOT_SUPPORTED'
            event = {'text':event_text,'type':fact_category,'attributes':[],
                'mention':event_text, 
                'mention_importance':fact_importance,
                'mention_classification': mention_classification }
    else:
        # if added_fact, accurate only for singleton events
        if (not pd.isna(fact_accuracy)) and (fact_accuracy=='accurate'):
            event = {'text':event_text,'type':fact_category,'attributes':[],
                        'mention':event_text,
                        'mention_importance':addedfact_importance,
                        'mention_classification': 'ENTAILED_BY' }
        else:
            #otherwise, it will be partial or it is just included to house the attribute addedfacts
            event = {'text':event_text,'type':fact_category,'attributes':[],
                        'mention':event_text,
                        'mention_importance': hallucination_harmfullness,
                        'mention_classification': 'OTHER_NOT_SUPPORTED' }
    return event


def update_event( event, row_cache, is_added_fact ) :
    """
    event is in form:
    { "text":"pelvic lymph node dissection in March, of 2018, shortly after his initial diagnosis",
      "type":"treatment_surgery",
      "mention":"dissection",
      "attributes":["pelvic lymph node", "March, of 2018", "after his initial diagnosis"] }
    """
    if len(row_cache) == 0 :
        if 'attribute_classifications' not in event:
            event['attribute_classifications'] = []
        if 'attribute_importances' not in event:
            event['attribute_importances'] = []
        return
    
    #default - this can happen for added facts
    mention=None
    mention_classification=None
    mention_importance=None

    attributes = []
    attribute_classifications = []
    attribute_importances = []

    for row in row_cache :
        event_argument = row['Unnamed: 2']
        fact_category = row['Unnamed: 3']

        #original_rubric
        fact_importance = row['Unnamed: 4']
        fact_presence = row['Unnamed: 6']
        #added-could be truth hallucination or true information not in rubric
        fact_accuracy = row['Unnamed: 7']
        addedfact_importance = row['Unnamed: 8']
        hallucination_harmfullness = row['Unnamed: 9']

        if not is_added_fact :
            # mention is case when it is split out
            if fact_category=='mention':
                mention = event_argument
                mention_classification = 'ENTAILED_BY' if fact_presence=='present' else 'OTHER_NOT_SUPPORTED'
                mention_importance=fact_importance
            else :
                attributes.append(event_argument)
                attribute_importances.append(fact_importance)
                if fact_presence=='present':
                    attribute_classifications.append('ENTAILED_BY')
                else:
                    attribute_classifications.append('OTHER_NOT_SUPPORTED')
        else :
            attributes.append(event_argument)
            if fact_accuracy=='accurate':
                attribute_classifications.append('ENTAILED_BY')
                attribute_importances.append(addedfact_importance)
            else:
                attribute_classifications.append('OTHER_NOT_SUPPORTED')
                attribute_importances.append(hallucination_harmfullness)
    
    event['mention']=mention
    event['mention_classification']=mention_classification
    event['mention_importance']=mention_importance
    event['attributes']=attributes
    event['attribute_classifications']=attribute_classifications
    event['attribute_importances']=attribute_importances

    return event


def validate_gradedrubric_sheet( df_sheet ) :
    """
    This tests for some general formatting issues based on the excel sheet rubric grading.
    """
    issues = []

    #check if the row headers exists
    if len( df_sheet[ df_sheet['RUBRIC CREATION']=='rubric_criterion' ] ) != 1:
        print('RUBRIC CREATION header either missing or there is multiple')
        issues.append('RUBRIC CREATION header either missing or there is multiple')
        return issues
    if len( df_sheet[ df_sheet[ 'RUBRIC CREATION']=='[ADDED INFORMATION  - in rubric grading]' ] ) != 1:
        print('[ADDED INFORMATION  - in rubric grading] either missing or there is multiple')
        issues.append('[ADDED INFORMATION  - in rubric grading] either missing or there is multiple')
        return issues
    
    #check if each fact_importance categories are correct
    start_rubric = df_sheet[ df_sheet['RUBRIC CREATION']=='rubric_criterion' ].index[0]
    start_addedinfo = df_sheet[ df_sheet['RUBRIC CREATION']=='[ADDED INFORMATION  - in rubric grading]' ].index[0]
    
    ##original rubric

    #fact category
    found_fact_category = df_sheet[ df_sheet.index>start_rubric ]['Unnamed: 3'].unique().tolist()
    found_fact_category = [ x.split(' ')[0] if isinstance(x,str) else x for x in found_fact_category ]
    invalid= set(found_fact_category) - set([pd.NA]+['mention','attribute']+FACT_CATEGORY)
    if len( invalid )==0 :
        print( 'found invalid fact_categories: {}'.format(invalid) )
        issues.append('fact-classification')

    #fact importance
    found_fact_importance = df_sheet[ df_sheet.index>start_rubric ]['Unnamed: 4'].unique().tolist()
    invalid = set(found_fact_importance) - set([pd.NA]+FACT_IMPORTANCE)
    if len( invalid )==0:
        print( 'found invalid fact_importance: {}'.format( invalid ) )
        issues.append('found invalid found_importance: {}'.format( invalid ) )

    ## grading

    #fact presence
    found_fact_presence = df_sheet[ df_sheet.index>start_rubric ]['Unnamed: 6'].unique().tolist()
    invalid = set(found_fact_presence) - set([pd.NA]+FACT_PRESENCE)
    if len( invalid ) ==0 :
        print( 'found invalid fact_presence: {}'.format( invalid ) )
        issues.append('found invalid fact_presence: {}'.format( invalid ) )

    #fact accuracy
    found_fact_accuracy = df_sheet[ df_sheet.index>start_rubric ]['Unnamed: 7'].unique().tolist()
    invalid = set(found_fact_accuracy) - set([pd.NA]+FACT_ACCURACY)
    if len( invalid ) ==0 :
        print( 'found invalid fact_accuracy: {}'.format( invalid ) )
        issues.append('found invalid fact_accuracy: {}'.format( invalid ) )
    
    #addedfact importance
    found_addedfact_importance = df_sheet[ df_sheet.index>start_rubric ]['Unnamed: 8'].unique().tolist()
    invalid = set(found_addedfact_importance) - set([pd.NA]+ADDEDFACT_CRITICALITY) 
    if len(invalid )==0 :
        print( 'found invalid addedfact_importance: {}'.format( invalid ) )
        issues.append('found_addedfact_importance: {}'.format( invalid ) )
    
    #hallucination severity
    found_hallucination_harmfullness= df_sheet[ df_sheet.index>start_rubric ]['Unnamed: 9'].unique().tolist()
    invalid = set(found_hallucination_harmfullness) - set([pd.NA]+HALLUCINATION_HARMFULNESS)
    if len( invalid )==0 :
        print( 'found invalid hallucination_harmfullness: {}'.format( invalid ) )
        issues.append('found invalid hallucination_harmfullness: {}'.format( invalid ) )

    #for original rubrics arguments (mention/attribute) - check fact_presence is filled
    event_tuples = []
    for index, row in df_sheet[ start_rubric+1:start_addedinfo ].iterrows() :
        event_text = row['Unnamed: 1']
        event_argument = row['Unnamed: 2']
        fact_category = row['Unnamed: 3']

        fact_importance = row['Unnamed: 4']
        fact_presence = row['Unnamed: 6']

        if not pd.isna( event_argument ) :
            invalid = set(found_fact_category) - set(FACT_CATEGORY)
            if len( invalid )==0 :
                print( 'event named but fact_classification not in the accepted list' )
                print( 'Index: ' + str(index)+ ' - ' + str(row) )
                issues.append('invalid fact_classification @[{}] - '.format(str(index),str(row)))
        
        if pd.isna( event_argument ) and pd.isna( fact_category ) :
            continue
        
        #start new event
        if not pd.isna( event_text ):
            event_tuples.append([])
        #add rows to event tuple
        event_tuples[-1].append( [event_text,event_argument,fact_category] )

        if ( not pd.isna( fact_importance ) ) and (pd.isna( fact_presence )) :
            issues.append('fact missing fact_presence @[{}] - {}'.format(index,str(row)))

    for event_tuple in event_tuples :
        if len(event_tuple)==1:
            continue
        num_mention = len( [ x[-1] for x in event_tuple if x[-1]=='mention' ] )
        if num_mention!=1:
            issues.append('# of mentions error: {} - {}'.format(event_tuple[0], str(event_tuple)))

    #added facts - check if accuracy is filled
    for index, row in df_sheet[ start_addedinfo+1: ].iterrows() :
        event_text = row['Unnamed: 1']
        event_argument = row['Unnamed: 2']
        fact_category = row['Unnamed: 3']
        fact_presence = row['Unnamed: 6']
        fact_accuracy = row['Unnamed: 7']
        addedfact_importance = row['Unnamed: 8']
        hallucination_harmfullness = row['Unnamed: 9']

        # if pd.isna( event_argument ) and pd.isna( fact_category ) :
        #     continue
        if pd.isna( event_argument ) and pd.isna( event_text ) :
            continue

        if ( not pd.isna( event_argument ) ) and pd.isna( fact_accuracy ) :
            issues.append('added fact missing fact_accuracy @[{}] - {}'.format(index, str(row)) )
        
        if ( not pd.isna( event_text ) ):
            if pd.isna(fact_presence) and pd.isna( fact_accuracy )  :
                issues.append('added fact missing fact_accuracy @[{}] - {}'.format(index, str(row)) )
            if (not pd.isna(fact_presence)) and ( fact_presence!='partially_present' ) :
                issues.append('added fact fact_presence not correct @[{}] - {}'.format(index, str(row)) )

        if ( fact_accuracy=='accurate' ) and  pd.isna( addedfact_importance ):
            issues.append('accurate, added fact missing addedfact_importance @[{}] - {}'.format(index, str(row)) )
        if ( fact_accuracy=='not_accurate' ) and  pd.isna( hallucination_harmfullness ):
            issues.append('not_accurate, added fact missing hallucination_harmfullness @[{}] - {}'.format(index, str(row)) )
    

    return issues


def parse_overallratings( df, sections=SECTIONS ) :
    section2results = {}
    for section in sections :
        section2results[section] = {
            'completeness': df[ df['Unnamed: 11']==section ]['Unnamed: 12'].iloc[0],
            'factual-accuracy': df[ df['Unnamed: 11']==section ]['Unnamed: 13'].iloc[0],
            'relevance': df[ df['Unnamed: 11']==section ]['Unnamed: 14'].iloc[0],
            'overall': df[ df['Unnamed: 11']==section ]['Unnamed: 15'].iloc[0],
            'comments': df[ df['Unnamed: 11']==section ]['Unnamed: 16'].iloc[0],
        }
    return section2results


def validate_ratings( ratings_by_section, sections=SECTIONS, rating_metrics=RATING_METRICS ) :
    issues = []
    for section in sections :
        if section not in ratings_by_section :
            issues.append('missing section: {}'.format(section))
        
        for rating_metric in rating_metrics :
            if rating_metric not in ratings_by_section[section] :
                issues.append('missing section:{}, rating_metric:{}'.format(section, rating_metric))
            
            rating = ratings_by_section[section][rating_metric]
            try:
                rating = int(rating)
            except:
                issues.append('ratings is not an integer: {}'.format(rating))
    return issues


if __name__ == "__main__":
    
    if len( sys.argv ) < 2 :
        sys.exit(0)
    fn = sys.argv[1]
    
    xls = pd.ExcelFile(fn)

    #read from excel sheet to a tbsummary
    df = pd.read_excel(fn,sheet_name=xls.sheet_names[0])
    tbsummary = TbSummary.read_from_sheet('CASE0001', 'AUTH001', df)
    print('Summary:')
    print( tbsummary.to_json() )

    #read from excel sheet to a tbgradedrubric
    df = pd.read_excel(fn,sheet_name=xls.sheet_names[-3])

    issues = validate_gradedrubric_sheet( df )
    print('Detected Issues:')
    print( issues )

    tbgradedrubric = TbGradedRubric.read_from_sheet('HYP_TEXT','REF_TEXT',df)
    print('Graded Rubric:')
    print( tbgradedrubric.to_json() )