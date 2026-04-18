from torch import nn
from torchvision import models

model_mapping = {
    "efficientnet_b0": (
        models.efficientnet_b0,
        {"weights": models.EfficientNet_B0_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "efficientnet_b1": (
        models.efficientnet_b1,
        {"weights": models.EfficientNet_B1_Weights.DEFAULT, "family": "efficientnet"},
    ),
    "densenet121": (
        models.densenet121,
        {"weights": models.DenseNet121_Weights.DEFAULT, "family": "densenet"},
    ),
    "densenet169": (
        models.densenet169,
        {"weights": models.DenseNet169_Weights.DEFAULT, "family": "densenet"},
    ),
    "resnet50": (
        models.resnet50,
        {"weights": models.ResNet50_Weights.IMAGENET1K_V2, "family": "resnet"},
    ),
}


class Model(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(Model, self).__init__()

        model_class, model_config = model_mapping[model_name]
        self.model = model_class(weights=model_config["weights"])

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self._get_in_features(model_config["family"])

        if model_config["family"] == "efficientnet":
            self.model.classifier = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "densenet":
            self.model.classifier = self._create_classifier(in_features, num_classes)
        elif model_config["family"] == "resnet":
            self.model.fc = self._create_classifier(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def _get_in_features(self, family: str) -> int:
        if family == "efficientnet":
            return self.model.classifier[1].in_features
        elif family == "densenet":
            return self.model.classifier.in_features
        elif family == "resnet":
            return self.model.fc.in_features

    def _create_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features // 2, num_classes),
        )


class ModelFactory:
    def __init__(self, name: str, num_classes: int):
        self.name = name
        self.num_classes = num_classes

    def __call__(self):
        if self.name not in model_mapping:
            valid_options = ", ".join(model_mapping.keys())
            raise ValueError(
                f"Invalid model name: '{self.name}'. Available options: {valid_options}"
            )

        return Model(self.name, self.num_classes)
