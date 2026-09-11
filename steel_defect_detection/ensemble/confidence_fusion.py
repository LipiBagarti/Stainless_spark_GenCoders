"""
Confidence fusion — combines detector ensemble confidence with
EfficientNet-B0 classifier confidence.

The final confidence is a weighted combination:
    final = detector_weight * detector_conf + classifier_weight * classifier_conf

Weights are configurable via config.yaml and should be tuned experimentally.
"""


def fuse_confidences(
    detector_confidence: float,
    classifier_confidence: float,
    detector_weight: float = 0.6,
    classifier_weight: float = 0.4,
) -> float:
    """
    Compute final confidence from detector and classifier scores.

    Args:
        detector_confidence: Confidence from the WBF-fused detector ensemble
        classifier_confidence: Confidence from EfficientNet-B0
        detector_weight: Weight for detector confidence (from config)
        classifier_weight: Weight for classifier confidence (from config)

    Returns:
        Combined confidence score
    """
    total_weight = detector_weight + classifier_weight
    if total_weight == 0:
        return detector_confidence

    final = (detector_weight * detector_confidence +
             classifier_weight * classifier_confidence) / total_weight
    return round(final, 4)
