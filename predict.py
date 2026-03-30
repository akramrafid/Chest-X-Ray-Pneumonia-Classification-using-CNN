import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import json
from pathlib import Path


IMAGE_SIZE = 128
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    class_names = checkpoint['class_names']
    model = PneumoniaCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model, class_names


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_single(image_path, model, class_names, transform):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    predicted_idx = probabilities.argmax()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx] * 100

    return {
        'image': str(image_path),
        'predicted_class': predicted_class,
        'confidence': round(float(confidence), 2),
        'probabilities': {class_names[i]: round(float(p) * 100, 2) for i, p in enumerate(probabilities)}
    }


def predict_directory(directory, model, class_names, transform):
    directory = Path(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [p for p in directory.rglob('*') if p.suffix.lower() in image_extensions]

    if not image_paths:
        print(f"No images found in {directory}")
        return []

    results = []
    print(f"Running inference on {len(image_paths)} images...")

    for path in image_paths:
        result = predict_single(path, model, class_names, transform)
        results.append(result)
        print(f"  {path.name}: {result['predicted_class']} ({result['confidence']:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Chest X-Ray Pneumonia Classifier Inference")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--directory", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/predictions.json")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading model from {args.checkpoint}")
    model, class_names = load_model(args.checkpoint)
    transform = get_transform()
    print(f"Model loaded. Classes: {class_names} | Device: {DEVICE}")

    results = []

    if args.image:
        result = predict_single(args.image, model, class_names, transform)
        results.append(result)
        print(f"\nImage: {result['image']}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Probabilities: {result['probabilities']}")

    elif args.directory:
        results = predict_directory(args.directory, model, class_names, transform)
        normal_count = sum(1 for r in results if r['predicted_class'] == 'NORMAL')
        pneumonia_count = sum(1 for r in results if r['predicted_class'] == 'PNEUMONIA')
        print(f"\nSummary: {normal_count} NORMAL | {pneumonia_count} PNEUMONIA")

    else:
        print("Provide --image or --directory. Example:")
        print("  python predict.py --image path/to/image.jpg")
        print("  python predict.py --directory path/to/folder/")
        return

    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()