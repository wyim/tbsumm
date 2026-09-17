import json
import copy
from scipy import stats

from tb_records import TbGradedRubric, ENTAILMENT_CLASSIFICATIONS


agents_2_max_completion_tokens = {
    'TBFactExtractor':3800,
    'TBFactContextEntailmentClassifier': 8000,
    'TBFactPatientHistoryEntailmentClassifier':  8000,
    'TBFactPatientHistoryEntailmentClassifierFix':  8000,
    'LLMAsJudge4Axes': 500,
    'LLMAsJudgeGeneric': 500,
}


def get_aoai_message_for_casesummary( pthx, tb_summary_prompts, agent, hx=50 ) :
    messages=[
        {
            "role": "system",
            "content": tb_summary_prompts[agent]['system_instructions'],
        },
        {
            "role": "user",
            "content": tb_summary_prompts[agent]['user_chat_prompt'],
        },
        {
            "role": "user",
            "content": json.dumps(pthx[-1*hx:]),
        }]
    return messages


def get_aoai_message_for_rating( tb_eval_prompts, summary_text, reference_text, agent='LLMAsJudge4Axes' ) :
    messages=[
        {
            "role": "system",
            "content": tb_eval_prompts[agent]['system_instructions'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_prompt'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_candidate'],
        },
        {
            "role": "user",
            "content": summary_text,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_reference'],
        },
        {
            "role": "user",
            "content": reference_text,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_output'],
        }
        ]
    return messages


def get_aoai_message_for_tbfactextraction( tb_eval_prompts, summary_text, agent='TBFactExtractor' ) :
    messages=[
        {
            "role": "system",
            "content": tb_eval_prompts[agent]['system_instructions'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_prompt'],
        },
        {
            "role": "user",
            "content": summary_text,
        }]
    return messages


def get_aoai_message_for_tbfactcontextentailclassifier( tb_eval_prompts, hypothesis, facts, context, agent='TBFactContextEntailmentClassifier' ) :
    messages=[
        {
            "role": "system",
            "content": tb_eval_prompts[agent]['system_instructions'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_prompt'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_sourcetextprelude'],
        },
        {
            "role": "user",
            "content": hypothesis,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_factlistprelude'],
        },
        {
            "role": "user",
            "content": facts,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_contextprelude'],
        },
        {
            "role": "user",
            "content": context,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_outputprelude'],
        },
    ]
    return messages



def get_aoai_message_for_tbfactpthxentailclassifier( tb_eval_prompts, hypothesis, facts, pthx, agent='TBFactPatientHistoryEntailmentClassifier', hx=50 ) :
    messages=[
        {
            "role": "system",
            "content": tb_eval_prompts[agent]['system_instructions'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_prompt'],
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_sourcetextprelude'],
        },
        {
            "role": "user",
            "content": hypothesis,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_factlistprelude'],
        },
        {
            "role": "user",
            "content": facts,
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_contextprelude'],
        },
        {
            "role": "user",
            "content": json.dumps(pthx[-1*hx:]),
        },
        {
            "role": "user",
            "content": tb_eval_prompts[agent]['user_chat_outputprelude'],
        },
    ]
    return messages


def get_aoai_message_for_jsonfix( json_str_to_fix ) :
    messages=[
        {
            "role": "system",
            "content": "You are a helpful data scientist assistant.",
        },
        {
            "role": "user",
            "content": "Given the following candidate json string, edit the string to be loadable in python code.",
        },
        {
            "role": "user",
            "content": 'JSON CANDIDATE:\n{}'.format(json_str_to_fix),
        },
        {
            "role": "user",
            "content": 'JSON:\n',
        },
    ]
    return messages


def fix_entailmentclassifications( entailmentclassification, default='OTHER_NOT_SUPPORTED' ) :
    """
    This makes sure the classifications are in list, if not gives default of not supported.

    Entailment classifications are a list of these objects:
    { "mention":"PSA", "attributes":["September, of 2021","detectable","0.31","starting to rise"],
          "mention_classification":"ENTAILED_BY", "mention_classification_reason": "This is entailed because there was indeed another PSA measurement after the other postoperative PSA",
          "attribute_classifications":["CONTRADICTED_BY","ENTAILED_BY","OTHER_NOT_SUPPORTED"], "attribute_classifications_reasons":["The PSA measurement after the initial postoperative PSA was in September 2020 according to the context text", "There was no mention of starting to rise PSA"] },
    """
    for fact in entailmentclassification :
        if ('mention_classification' not in fact) or (fact['mention_classification'] not in ENTAILMENT_CLASSIFICATIONS) :
            fact['mention_classification'] = default
        if 'attribute_classifications' not in fact :
            fact['attribute_classifications'] = []
        for ind in range( len( fact['attribute_classifications'] ) ) :
            attrclass = fact['attribute_classifications'][ind]
            fact['attribute_classifications'][ind] = attrclass if attrclass in ENTAILMENT_CLASSIFICATIONS else default


def convert_to_tbgradedrubric( entailmentclassifications, candidate_text=None, reference_text=None ) :
    """
    Entailment classifications are a list of these objects:
    { "mention":"PSA", "attributes":["September, of 2021","detectable","0.31","starting to rise"],
          "mention_classification":"ENTAILED_BY", "mention_classification_reason": "This is entailed because there was indeed another PSA measurement after the other postoperative PSA",
          "attribute_classifications":["CONTRADICTED_BY","ENTAILED_BY","OTHER_NOT_SUPPORTED"], "attribute_classifications_reasons":["The PSA measurement after the initial postoperative PSA was in September 2020 according to the context text", "There was no mention of starting to rise PSA"] },
    """
    is_original_rubricitem = [True]*len(entailmentclassifications)
    tbgradedrubric = TbGradedRubric(candidate_text,
                   reference_text,
                   copy.deepcopy( entailmentclassifications ),
                   is_original_rubricitem)
    return tbgradedrubric


def harmonize_classification( classification1, classification2 ) :
    return 'ENTAILED_BY' if classification2=='ENTAILED_BY' else classification1


def harmonize_entailmentclassifications( entailmentclassification1, entailmentclassification2 ) :
    """
    This is used for the case if you have the same facts which are being classified for entailement from different sources.
    The use case here is entailment against gold summary or against the patient history.

    Classification will go to ENTAILED_BY if entailmentclassification2 is entailed_by.
    Assumes length of entailmentclassification1 and entailmentclassification2 are the same.

    entailmentclassification1 will be changed.
    """
    for ind1, classification1 in enumerate( entailmentclassification1 ) :
        classification2 = entailmentclassification2[ind1]

        mention_classification1 = classification1['mention_classification']
        mention_classification2 = classification2['mention_classification']

        attribute_classifications1 = classification1['attribute_classifications']
        attribute_classifications2 = classification2['attribute_classifications']

        classification1['mention_classification'] = harmonize_classification( mention_classification1, mention_classification2 )

        for ind2, attribute_classification1 in enumerate( attribute_classifications1 ) :
            attribute_classification2 = attribute_classifications2[ind2]
            attribute_classifications1[ind2] = harmonize_classification( attribute_classification1,
                                                                         attribute_classification2 )


def get_rubric_score( tbgradedrubric, error_weight=-0.25 ) :
    """
    For original rubric, is_original_rubric will be true for those coming from gold
    """
    # 1 if True, 0 otherwise
    additional_weights = [ 1 if x else 0 for x in tbgradedrubric.is_original_rubric ]
    return tbgradedrubric.score_directional( additional_weights=additional_weights, error_weight=error_weight )


def get_rubric_score_precision( tbgradedrubric, error_weight=-0.25 ) :
    """
    For original rubric, is_original_rubric will be true for those coming from gold.
    This scoring tries to mimic precision based system - so only TP and FP will count.
    """
    # first add 1 if not original
    additional_weights = [ 0 if x else 1 for x in tbgradedrubric.is_original_rubric ]
    # add back parts of the rubric that were entailed (for true positives)
    for ind, fact in enumerate( tbgradedrubric.facts ) :
        if fact['mention_classification'] == 'ENTAILED_BY':
            additional_weights[ind] = 1
    return tbgradedrubric.score_directional( additional_weights=additional_weights, error_weight=error_weight )


def get_rubric_f1( tbgradedrubric, error_weight=-0.25 ) :
    """
    Calculate f1 score for rubric. This assumes the is_original_rubric marks the gold; everything else is facts by the system.
    """
    rubric_recall = get_rubric_score( tbgradedrubric, error_weight=error_weight )
    rubric_precision = get_rubric_score_precision( tbgradedrubric, error_weight=error_weight )

    den = rubric_recall + rubric_precision
    f1 = 2*rubric_recall*rubric_precision/den if den>0 else 0.0

    return f1


def get_tbfact_score( tbgradedrubric_gold, tbgradedrubric_sys, error_weight=-0.25  ) :
    """
    TBFactv1.0 - no punishment
    TBFactv2.0 - deduct 0.25 for contradictory facts
    """
    tbfact_recall = tbgradedrubric_gold.score_directional( additional_weights=None, error_weight=error_weight )
    tbfact_precision = tbgradedrubric_sys.score_directional( additional_weights=None, error_weight=error_weight )

    den = tbfact_recall + tbfact_precision
    f1 = 2*tbfact_recall*tbfact_precision/den if den>0 else 0.0

    return f1


def get_correlations(x,y) :
    kendalltau, k_pval = stats.kendalltau(x, y)
    pearson, p_pval = stats.pearsonr(x, y)
    spearman, s_pval = stats.spearmanr(x, y)
    return kendalltau, pearson, spearman, k_pval, p_pval, s_pval