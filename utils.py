import numpy as np
import xml.etree.ElementTree as ET


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


def convert_copa(file_path):
    """
    Converting Choice of Plausible Alternatives (COPA)-formatted data to multiple formats for following tasks:
    - sequence classification
    - multiple choice

    :param file_path: full path of the file we want to convert the samples from
    Current list of the supported files:
    - copa-all.xml: all samples (dev+test) from COPA
    - copa-dev.xml: only dev samples from COPA
    - copa-test.xml: only test samples from COPA
    - BCOPA-CE.xml: all samples (1000) of BCOPA-CE (this data is only for TEST, neither train nor dev)
    - balanced-copa-dev-all.xml: all samples (1000) of BCOPA (this data is only for TRAIN/DEV)
    :return:
    - data: in data, "cause" is NOT necessarily always the first argument
    - data_cause_effect: here, "cause" IS always the first argument. (order: cause-effect)
    - data_multi_choice: for each instance, we have two records: one with premise and the correct hypothesis and
    the other one with premise and the incorrect hypothesis.
    """

    # in data_cause_effect, we assume the first and second arguments are cause and effect, respectively.
    # and of course, there always can be wrong cause and effect, but the order does not change
    data = []
    data_cause_effect = []
    data_multi_choice = []

    def create_record(r_id, premise, hypothesis, label):
        return {'id': r_id, 'sent1': premise, 'sent2': hypothesis, 'label': label}

    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file_path, parser=parser)
        root = tree.getroot()

        # each item has (Premise, Hypothesis_1, Hypothesis_2) and a question type (asks-for)
        for item in root.findall("./item"):
            # item[0]: premise
            # item[1]: hypothesis 1
            # item[2]: hypothesis 2
            spans = {0: item[0].text, 1: item[1].text, 2: item[2].text}

            span1_premise = spans[0]
            span2_correct = spans[int(item.attrib["most-plausible-alternative"])]
            span2_incorrect = spans[2] if int(item.attrib["most-plausible-alternative"]) == 1 else spans[1]

            # ----------------------------------------------------------------------------------------
            # for data, we ignore the order meaning the first argument could be either cause or effect
            data.append(create_record('{}-1'.format(str(item.attrib["id"])), span1_premise, span2_correct, 1))
            data.append(create_record('{}-2'.format(str(item.attrib["id"])), span1_premise, span2_incorrect, 0))

            # ----------------------------------------------------------------------------------------
            # for data_cause_effect, cause (either right or wrong cause) is always first
            if item.attrib["asks-for"] == "cause":
                data_cause_effect.append(
                    create_record('{}-1'.format(str(item.attrib["id"])), span2_correct, span1_premise, 1))
                data_cause_effect.append(
                    create_record('{}-2'.format(str(item.attrib["id"])), span2_incorrect, span1_premise, 0))
            elif item.attrib["asks-for"] == "effect":
                data_cause_effect.append(
                    create_record('{}-1'.format(str(item.attrib["id"])), span1_premise, span2_correct, 1))
                data_cause_effect.append(
                    create_record('{}-2'.format(str(item.attrib["id"])), span1_premise, span2_incorrect, 0))

            # ----------------------------------------------------------------------------------------
            # data_multi_choice
            label = int(item.attrib["most-plausible-alternative"])
            data_multi_choice.append({'id': item.attrib["id"],
                                      'premise': spans[0],
                                      'question': 'What is the {}?'.format(item.attrib["asks-for"]),
                                      'choice0': spans[1],
                                      'choice1': spans[2],
                                      'label': label - 1})

    except Exception as e:
        print("[crest-log] copa2bert. Detail: {}".format(e))

    return data, data_cause_effect, data_multi_choice
