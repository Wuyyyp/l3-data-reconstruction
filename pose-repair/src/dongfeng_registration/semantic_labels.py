from __future__ import annotations


DEFAULT_DYNAMIC_CLASSES = frozenset(
    {
        "person",
        "rider",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "trailer",
        "caravan",
        "animal",
        "vegetation",
        "sky",
    }
)

LABEL_ALIASES = {
    "construction--flat--road-marking": "road_marking",
    "human--person": "person",
    "human--rider": "rider",
    "nature--vegetation": "vegetation",
    "animal--ground-animal": "animal",
    "object--vehicle--bicycle": "bicycle",
    "object--vehicle--bus": "bus",
    "object--vehicle--caravan": "caravan",
    "object--vehicle--car": "car",
    "object--vehicle--motorcycle": "motorcycle",
    "object--vehicle--trailer": "trailer",
    "object--vehicle--train": "train",
    "object--vehicle--truck": "truck",
}


def canonicalize_label(label: str) -> str:
    value = str(label).strip().lower().replace(" ", "_")
    if value in LABEL_ALIASES:
        return LABEL_ALIASES[value]
    if "vegetation" in value:
        return "vegetation"
    if value.endswith("person") or "person" in value:
        return "person"
    if value.endswith("rider") or "rider" in value:
        return "rider"
    if value.endswith("bicycle") or "bicycle" in value:
        return "bicycle"
    if value.endswith("motorcycle") or "motorcycle" in value:
        return "motorcycle"
    if value.endswith("truck") or "truck" in value:
        return "truck"
    if value.endswith("train") or "train" in value:
        return "train"
    if value.endswith("trailer") or "trailer" in value:
        return "trailer"
    if value.endswith("caravan") or "caravan" in value:
        return "caravan"
    if value.endswith("animal") or "animal" in value:
        return "animal"
    if value.endswith("bus") or "bus" in value:
        return "bus"
    if value.endswith("car") or "car" in value:
        return "car"
    return value
