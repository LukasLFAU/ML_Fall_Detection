# Model Evaluation

## Evaluation Setup

The fall detection model was evaluated using three different evaluation settings. The first evaluation uses a random recording-level train/test split. This means that recordings from both participants can appear in both the training and test set. This split is useful as a first baseline, but it can be overly optimistic because similar movement patterns from the same persons may be present in both sets.

To better assess generalization, two additional person-holdout evaluations were performed. In these evaluations, the model is trained on one participant and tested on the other participant. This is a stricter setup because the model has to classify recordings from a person it has not seen during training.

## Evaluation 1: Random Recording-Level Split

In the random recording-level split, the model correctly classified all test recordings. The confusion matrix shows 25 correctly detected fall recordings and 36 correctly detected non-fall recordings, with no false positives and no false negatives.

While this result looks perfect, it should be interpreted carefully. Since recordings from both participants are mixed across training and test data, the model may benefit from person-specific movement patterns and similar recording conditions.

## Evaluation 2: Train on Lukas, Test on Polina

In this person-holdout evaluation, the model was trained only on Lukas' recordings and tested on Polina's recordings. The model correctly classified 18 fall recordings and 57 non-fall recordings. However, 26 fall recordings were incorrectly classified as non-fall, while only 2 non-fall recordings were incorrectly classified as fall.

This means that the model was rather conservative when predicting falls for Polina. When it predicted a fall, it was usually correct, but it missed many actual fall recordings. This is reflected in a high fall precision but a lower fall recall.

## Evaluation 3: Train on Polina, Test on Lukas

In the second person-holdout evaluation, the model was trained only on Polina's recordings and tested on Lukas' recordings. The model correctly classified 39 fall recordings and 39 non-fall recordings. Only 1 fall recording was missed, but 21 non-fall recordings were incorrectly classified as fall.

This means that the model detected almost all fall recordings from Lukas, but it also produced more false alarms. This is reflected in a high fall recall but lower fall precision.

## Interpretation

The results show that the model can clearly distinguish fall and non-fall recordings in a random recording-level split. However, the person-holdout evaluations show that generalization to unseen participants is more difficult. This indicates that the model may partly learn person-specific movement patterns or recording-specific characteristics.

For this reason, the model should be interpreted as a prototype trained on a small, controlled dataset. The results are promising for the collected dataset, but they are not sufficient to claim robust real-world fall detection performance. A larger dataset with more participants, more recording sessions, different smartphone positions and more diverse non-fall activities would be needed to improve generalization.
