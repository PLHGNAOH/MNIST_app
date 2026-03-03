import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms


# =========================
# Model Definition
# =========================
class LeNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # padding=2 tương đương "same" với kernel=5
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgpool2 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.avgpool1(self.conv1(x)))
        x = F.relu(self.avgpool2(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model(model_path):
    model = LeNetClassifier(num_classes=10)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


model = load_model("lenet_model.pt")


# =========================
# Inference
# =========================
def inference(image, model):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    return confidence.item() * 100, pred.item()


# =========================
# Streamlit UI
# =========================
def main():
    st.title("Digit Recognition")
    st.subheader("Model: LeNet - Dataset: MNIST")

    option = st.selectbox(
        "Choose input method:",
        ("Upload Image File", "Run Example Image")
    )

    if option == "Upload Image File":
        file = st.file_uploader(
            "Upload an image of a digit",
            type=["jpg", "png"]
        )

        if file is not None:
            image = Image.open(file)
            prob, label = inference(image, model)

            st.image(image, caption="Input Image")
            st.success(
                f"Prediction: {label} ({prob:.2f}% confidence)"
            )

    else:
        try:
            image = Image.open("demo_8.png")
            prob, label = inference(image, model)

            st.image(image, caption="Example Image")
            st.success(
                f"Prediction: {label} ({prob:.2f}% confidence)"
            )
        except:
            st.warning("demo_8.png not found in repository.")


if __name__ == "__main__":
    main()