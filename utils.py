import numpy as np


def lower_nth(s, n):
    return s[:n] + s[n].lower() + s[n + 1:]


def capitalize_nth(s, n):
    return s[:n] + s[n].capitalize() + s[n + 1:]


def compute_metrics(eval_predictions):
    # predictions, label_ids = eval_predictions
    predictions = eval_predictions.predictions[0] if isinstance(eval_predictions.predictions,
                                                                tuple) else eval_predictions.predictions
    label_ids = eval_predictions.label_ids
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def get_atomic_relation_map(return_modified_templates=False):
    if not return_modified_templates:
        return {
            'AtLocation': 'located or found at/in/on',
            'CapableOf': 'is/are capable of',
            'Causes': 'causes',
            'CausesDesire': 'makes someone want',
            'CreatedBy': 'is created by',
            'Desires': 'desires',
            'HasA': 'has, possesses or contains',
            'HasFirstSubevent': 'BEGINS with the event/action',
            'HasLastSubevent': 'ENDS with the event/action',
            'HasPrerequisite': 'to do this, one requires',
            'HasProperty': 'can be characterized by being/having',
            'HasSubEvent': 'includes the event/action',
            'HinderedBy': 'can be hindered by',
            'InstanceOf': 'is an example/instance of',
            'isAfter': 'happens after',
            'isBefore': 'happens before',
            'isFilledBy': 'blank can be filled by',
            'MadeOf': 'is made of',
            'MadeUpOf': 'made (up) of',
            'MotivatedByGoal': 'is a step towards accomplishing the goal',
            'NotDesires': 'do(es) NOT desire',
            'ObjectUse': 'used for',
            'UsedFor': 'used for',
            'oEffect': 'as a result, Y or others will',
            'oReact': 'as a result, Y or others feels',
            'oWant': 'as a result, Y or others want',
            'PartOf': 'is a part of',
            'ReceivesAction': 'can receive or be affected by the action',
            'xAttr': 'X is seen as',
            'xEffect': 'as a result, PersonX will',
            'xIntent': 'because PersonX wanted',
            'xNeed': 'but before, PersonX needed',
            'xReact': 'as a result, PersonX feels',
            'xReason': 'because',
            'xWant': 'as a result, PersonX wants'
        }
    else:
        # every key has a value of type `list` with the following fields (`triple` is a knowledge graph entry like:
        # [subject, relation, target]) that indicates a relation in ATOMIC 2020 knowledge graph
        # 1) string: human-readable description of relation in the triple
        # 2) int: 0 or 1 indicating whether the relation is part of the first sentence with subject or should be
        # an independent sentence together with the target in the triple, respectively.
        # 3) category of relation defined by the ATOMIC paper: physical, social, event
        return {
            'AtLocation': ['located or found at/in/on', 0, 'physical'],
            'CapableOf': ['is/are capable of', 0, 'physical'],
            'Causes': ['causes', 0, 'event'],
            'CausesDesire': ['makes someone want', 0, 'social'],
            'CreatedBy': ['is created by', 0, ''],
            'Desires': ['desires', 0, 'physical'],
            'HasA': ['has, possesses or contains', 0, 'physical'],
            'HasFirstSubevent': ['begins with the event/action', 0, 'event'],
            'HasLastSubevent': ['ends with the event/action', 0, 'event'],
            'HasPrerequisite': ['To do this, one requires', 1, 'social'],
            'HasProperty': ['can be characterized by being/having', 0, 'physical'],
            'HasSubEvent': ['includes the event/action', 0, 'event'],
            'HinderedBy': ['can be hindered by', 0, 'event'],
            'InstanceOf': ['is an example or instance of', 0, ''],
            'isAfter': ['happens after', 0, 'event'],
            'isBefore': ['happens before', 0, 'event'],
            'isFilledBy': ['blank can be filled by', 0, 'event'],
            'MadeOf': ['is made of', 0, 'physical'],
            'MadeUpOf': ['made or made up of', 0, 'physical'],
            'MotivatedByGoal': ['is a step towards accomplishing the goal', 0, 'social'],
            'NotDesires': ['do not or does not desire', 0, 'physical'],
            'ObjectUse': ['used for', 0, 'physical'],
            'UsedFor': ['used for', 0, 'physical'],
            'oEffect': ['As a result, PersonY or others will', 1, 'social'],
            'oReact': ['As a result, PersonY or others feels', 1, 'social'],
            'oWant': ['As a result, PersonY or others want', 1, 'social'],
            'PartOf': ['is a part of', 0, 'physical'],
            'ReceivesAction': ['can receive or be affected by the action', 0, ''],
            'xAttr': ['X is seen as', 1, 'social'],
            'xEffect': ['As a result, PersonX will', 1, 'social'],
            'xIntent': ['because PersonX wanted', 0, 'social'],
            'xNeed': ['but before, PersonX needed', 0, 'social'],
            'xReact': ['As a result, PersonX feels', 1, 'social'],
            'xReason': ['because', 0, 'event'],
            'xWant': ['As a result, PersonX wants', 1, 'social']
        }
