import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from typing import TypedDict


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "modele"
DETAILS = re.compile(r"\([^)]*\)|<[^>]*>|\$[^$]*\$")
TIME_MARKER = re.compile(r"t(\d+)", re.IGNORECASE)
RANDOM_STATE = 42
ALPHA = 0.01
VALIDATION_SIZE = 0.25
MODEL_FEATURE_PREFIXES = (
    "navigateur=",
    "detail=",
    "transition=",
    "detail_transition=",
    "triple=",
    "first_detail_",
    "last_detail_",
    "first_action_",
    "last_action_",
)


class NaiveBayesModel(TypedDict):
    classes: list[str]
    log_priors: dict[str, float]
    feature_deltas: dict[str, list[tuple[str, float]]]
    log_denominators: dict[str, float]
    vocabulary: set[str]


def normalize_action(value: str) -> str:
    action = DETAILS.sub("", value).strip()
    if action.endswith("1"):
        action = action[:-1].rstrip()
    return action


def extract_features(row: list[str], training: bool) -> tuple[dict[str, float], str | None]:
    user = row[0].strip() if training else None
    browser = row[1].strip() if training else row[0].strip()
    events = row[2:] if training else row[1:]
    features: Counter[str] = Counter()
    features[f"navigateur={browser}"] = 1
    actions: list[str] = []
    detailed_events: list[str] = []
    duration = 0

    for raw_value in events:
        value = raw_value.strip()
        if not value:
            continue
        time_match = TIME_MARKER.fullmatch(value)
        if time_match:
            duration = max(duration, int(time_match.group(1)))
            continue

        action = normalize_action(value)
        if not action:
            continue
        actions.append(action)
        detailed_events.append(value)
        features[f"action={action}"] += 1
        features[f"detail={value}"] += 1

    for previous, current in zip(actions, actions[1:]):
        features[f"transition={previous} -> {current}"] += 1

    for previous, current in zip(detailed_events, detailed_events[1:]):
        features[f"detail_transition={previous} -> {current}"] = 1.0

    for first, second, third in zip(actions, actions[1:], actions[2:]):
        features[f"triple={first} -> {second} -> {third}"] = 1.0

    for position, value in enumerate(detailed_events[:5]):
        features[f"first_detail_{position}={value}"] = 1.0
    for position, value in enumerate(actions[:5]):
        features[f"first_action_{position}={value}"] = 1.0
    for offset, value in enumerate(reversed(detailed_events[-5:])):
        features[f"last_detail_{offset}={value}"] = 1.0
    for offset, value in enumerate(reversed(actions[-5:])):
        features[f"last_action_{offset}={value}"] = 1.0

    features["log_nombre_actions"] = math.log1p(len(actions))
    features["log_duree_minutes"] = math.log1p(duration / 60)
    return dict(features), user


def load_dataset(filename: str, training: bool) -> tuple[list[dict[str, float]], list[str]]:
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    features: list[dict[str, float]] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            minimum = 2 if training else 1
            if len(row) < minimum:
                raise ValueError(
                    f"Ligne {row_number} invalide dans {path} : "
                    f"{minimum} champs minimum attendus."
                )
            row_features, user = extract_features(row, training)
            features.append(row_features)
            if training:
                if user is None:
                    raise ValueError(f"Identifiant utilisateur absent ligne {row_number}.")
                labels.append(user)
    return features, labels


def stratified_split(labels: list[str]) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)

    generator = random.Random(RANDOM_STATE)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError(
                "La validation nécessite au moins deux sessions par utilisateur."
            )
        generator.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * VALIDATION_SIZE)),
        )
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return train_indices, validation_indices


def fit_naive_bayes(
    documents: list[dict[str, float]],
    labels: list[str],
) -> NaiveBayesModel:
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(
            feature
            for feature in document
            if feature.startswith(MODEL_FEATURE_PREFIXES)
        )

    vocabulary = {
        feature
        for feature, frequency in document_frequency.items()
        if frequency >= 2 and frequency <= len(documents) * 0.9
    }
    class_documents: Counter[str] = Counter()
    class_features: dict[str, Counter[str]] = {}
    for document, label in zip(documents, labels):
        class_documents[label] += 1
        feature_counts = class_features.setdefault(label, Counter())
        feature_counts.update(
            feature for feature in document if feature in vocabulary
        )

    classes = sorted(class_documents)
    denominators = {
        label: sum(class_features[label].values()) + ALPHA * len(vocabulary)
        for label in classes
    }
    log_denominators = {
        label: math.log(denominators[label]) for label in classes
    }
    log_priors = {
        label: math.log(class_documents[label] / len(labels))
        for label in classes
    }
    feature_deltas: dict[str, list[tuple[str, float]]] = {}
    for label in classes:
        for feature, count in class_features[label].items():
            if count:
                feature_deltas.setdefault(feature, []).append(
                    (label, math.log((count + ALPHA) / ALPHA))
                )

    return {
        "classes": classes,
        "log_priors": log_priors,
        "feature_deltas": feature_deltas,
        "log_denominators": log_denominators,
        "vocabulary": vocabulary,
    }


def predict_naive_bayes(
    model: NaiveBayesModel,
    documents: list[dict[str, float]],
) -> list[str]:
    classes = model["classes"]
    log_priors = model["log_priors"]
    feature_deltas = model["feature_deltas"]
    log_denominators = model["log_denominators"]
    vocabulary = model["vocabulary"]
    predictions = []

    for document in documents:
        known_features = {
            feature: 1.0
            for feature in document
            if feature in vocabulary
        }
        total_weight = sum(known_features.values())
        scores = {
            label: log_priors[label]
            + total_weight * (math.log(ALPHA) - log_denominators[label])
            for label in classes
        }
        for feature, value in known_features.items():
            for label, delta in feature_deltas.get(feature, ()):
                scores[label] += value * delta
        predictions.append(max(classes, key=scores.__getitem__))
    return predictions


def classification_scores(expected: list[str], predicted: list[str]) -> dict[str, float]:
    correct = sum(actual == guess for actual, guess in zip(expected, predicted))
    labels = set(expected)
    per_class_f1 = []
    for label in labels:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(
            2 * true_positive / denominator if denominator else 0.0
        )
    return {
        "accuracy": round(correct / len(expected), 4),
        "f1_macro": round(sum(per_class_f1) / len(per_class_f1), 4),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_features, labels = load_dataset("train.csv", training=True)
    test_features, _ = load_dataset("test.csv", training=False)

    label_counts = Counter(labels)
    train_indices, validation_indices = stratified_split(labels)
    validation_set = set(validation_indices)
    split_path = OUTPUT_DIR / "validation_split.csv"
    with split_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["session_train", "utilisateur", "role"])
        writer.writerows(
            (
                index + 1,
                user,
                "validation" if index in validation_set else "apprentissage",
            )
            for index, user in enumerate(labels)
        )

    x_train = [train_features[index] for index in train_indices]
    x_validation = [train_features[index] for index in validation_indices]
    y_train = [labels[index] for index in train_indices]
    y_validation = [labels[index] for index in validation_indices]

    model = fit_naive_bayes(x_train, y_train)
    validation_predictions = predict_naive_bayes(model, x_validation)
    model_results = classification_scores(y_validation, validation_predictions)
    majority_label = Counter(y_train).most_common(1)[0][0]
    baseline_results = classification_scores(
        y_validation,
        [majority_label] * len(y_validation),
    )

    metrics = {
        "validation": {
            "methode": "separation unique stratifiee",
            "random_state": RANDOM_STATE,
            "part_validation": VALIDATION_SIZE,
            "sessions_entrainement": len(train_indices),
            "sessions_validation": len(validation_indices),
            "utilisateurs": len(label_counts),
            "nombre_features": len(model["vocabulary"]),
            "resultats": {
                "Bayes naïf présence binaire": model_results,
                "Majoritaire (baseline)": baseline_results,
            },
            "modele": "Bayes naïf présence binaire",
        }
    }
    print("Validation stratifiée sur une seule séparation :")
    print(f"Utilisateurs : {len(label_counts)}")
    print(
        f"Sessions : {len(train_indices)} pour apprendre, "
        f"{len(validation_indices)} pour valider"
    )
    print(
        "- Bayes naïf présence binaire : "
        f"accuracy={model_results['accuracy']:.4f}, "
        f"F1 macro={model_results['f1_macro']:.4f}"
    )
    print(
        "- Baseline majoritaire : "
        f"accuracy={baseline_results['accuracy']:.4f}, "
        f"F1 macro={baseline_results['f1_macro']:.4f}"
    )
    print("Modèle : Bayes naïf présence binaire")

    final_model = fit_naive_bayes(train_features, labels)
    test_predictions = predict_naive_bayes(final_model, test_features)

    submission_path = OUTPUT_DIR / "submission_baseline.csv"
    with submission_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["RowId", "prediction"])
        writer.writerows(
            (row_id, prediction)
            for row_id, prediction in enumerate(test_predictions, start=1)
        )

    metrics["test"] = {
        "sessions_predites": len(test_predictions),
        "utilisateurs_predits_distincts": len(set(test_predictions)),
        "distribution_predictions": dict(Counter(test_predictions).most_common()),
    }
    metrics_path = OUTPUT_DIR / "evaluation.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prédictions de travail : {submission_path}")
    print(f"Affectation apprentissage/validation : {split_path}")
    print(f"Mesures et distribution des prédictions : {metrics_path}")


if __name__ == "__main__":
    main()
