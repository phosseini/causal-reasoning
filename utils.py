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


def get_atomic_relation_templates():
    # every key has a value of type `list` with the following fields (`triple` is a knowledge graph entry like:
    # [subject, relation, target]) that indicates a relation in ATOMIC 2020 knowledge graph
    # 1) string: original human-readable description of relations provided by: https://arxiv.org/abs/2010.05953
    # 2) string: [maybe] modified human-readable description of relations
    # 3) int: 0 or 1 indicating whether the relation is part of the first sentence with subject or should be
    # an independent sentence together with the target in the triple, respectively.
    # 4) category of relation defined by the ATOMIC paper: physical, social, event
    return {
        'AtLocation': ['located or found at/in/on', 'located or found at', 0, 'physical'],
        'CapableOf': ['is/are capable of', 'is capable of', 0, 'physical'],
        'Causes': ['causes', 'causes', 0, 'event'],
        'CausesDesire': ['makes someone want', 'makes someone want', 0, 'social'],
        'CreatedBy': ['is created by', 'is created by', 0, ''],
        'Desires': ['desires', 'desires', 0, 'physical'],
        'HasA': ['has, possesses or contains', 'has, possesses or contains', 0, 'physical'],
        'HasFirstSubevent': ['BEGINS with the event/action', 'begins with', 0, 'event'],
        'HasLastSubevent': ['ENDS with the event/action', 'ends with', 0, 'event'],
        'HasPrerequisite': ['to do this, one requires', 'requires', 1, 'social'],
        'HasProperty': ['can be characterized by being/having', 'can be characterized by', 0, 'physical'],
        'HasSubEvent': ['includes the event/action', 'includes', 0, 'event'],
        'HinderedBy': ['can be hindered by', 'can be hindered by', 0, 'event'],
        'InstanceOf': ['is an example/instance of', 'is an example of', 0, ''],
        'isAfter': ['happens after', 'happens after', 0, 'event'],
        'isBefore': ['happens before', 'happens before', 0, 'event'],
        'isFilledBy': ['blank can be filled by', 'can be filled by', 0, 'event'],
        'MadeOf': ['is made of', 'is made of', 0, 'physical'],
        'MadeUpOf': ['made (up) of', 'is made up of', 0, 'physical'],
        'MotivatedByGoal': ['is a step towards accomplishing the goal', 'is a step towards accomplishing', 0, 'social'],
        'NotDesires': ['do(es) NOT desire', 'does not desire', 0, 'physical'],
        'ObjectUse': ['used for', 'is used for', 0, 'physical'],
        'UsedFor': ['used for', 'is used for', 0, 'physical'],
        'oEffect': ['as a result, Y or others will', 'As a result, PersonY', 1, 'social'],
        'oReact': ['as a result, Y or others feels', 'As a result, PersonY feels', 1, 'social'],
        'oWant': ['as a result, Y or others want', 'As a result, PersonY wants', 1, 'social'],
        'PartOf': ['is a part of', 'is a part of', 0, 'physical'],
        'ReceivesAction': ['can receive or be affected by the action', 'can receive or be affected by', 0, ''],
        'xAttr': ['X is seen as', 'PersonX is seen as', 1, 'social'],
        'xEffect': ['as a result, PersonX will', 'As a result, PersonX', 1, 'social'],
        'xIntent': ['because PersonX wanted', 'because PersonX wanted', 0, 'social'],
        'xNeed': ['but before, PersonX needed', 'but before, PersonX needed', 0, 'social'],
        'xReact': ['as a result, PersonX feels', 'As a result, PersonX feels', 1, 'social'],
        'xReason': ['because', 'because', 0, 'event'],
        'xWant': ['as a result, PersonX wants', 'As a result, PersonX wants', 1, 'social']
    }


def convert_copa_data(file_path):
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
        print("[copa-convert-log] error detail: {}".format(e))

    return data, data_cause_effect, data_multi_choice


def longest_prefix(a, b):
    prefix = ""
    a = a.split()
    b = b.split()
    min_len = min(len(a), len(b))
    i = 0
    while i < min_len:
        if a[i] == b[i]:
            prefix += a[i] + " "
        else:
            break
        i += 1
    return prefix.strip()
