class LabelEncoder:
    def __init__(self):
        self.label_map = {}

    def fit(self, labels):
        unique_labels = set(labels)
        for i, label in enumerate(unique_labels):
            self.label_map[label] = i
            print(f'"{label}" is encoded as {i}')

    def transform(self, labels):
        transformed_labels = []
        for label in labels:
            transformed_labels.append(self.label_map[label])
        return transformed_labels