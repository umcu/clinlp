"""Code for converting the Mantra GSC corpus to json format readable by clinlp."""

import itertools
import json
import pathlib
import xml.etree.ElementTree as ElementTree

if __name__ == "__main__":
    medline = ElementTree.parse("Medline_GSC_nl_man.xml").getroot()  # noqa S314
    emea = ElementTree.parse("EMEA_GSC_nl_man.xml").getroot()  # noqa S314

    docs = []

    for doc in itertools.chain(medline, emea):
        doc_id = doc.attrib["id"]

        text = None
        annotations = []

        for child in doc[0]:
            if child.tag == "text":
                text = child.text
            elif child.tag == "e":
                annotations.append(
                    {
                        "text": child.text,
                        "start": int(child.attrib["offset"]),
                        "end": int(child.attrib["offset"]) + int(child.attrib["len"]),
                        "label": child.attrib["cui"],
                    }
                )

        docs.append({"identifier": doc_id, "text": text, "annotations": annotations})

    output = {"docs": docs}

    with pathlib.Path("data/mantra_gsc.json").open("w") as f:
        json.dump(output, f, indent=4)
