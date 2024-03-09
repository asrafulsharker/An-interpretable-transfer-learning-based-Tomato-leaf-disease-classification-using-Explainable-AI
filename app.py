from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from datetime import datetime

app = Flask(__name__)

num_classes = 11  # Replace this with the actual number of classes

class_info = {
    0: ("Late blight", "Late blight is a destructive fungal disease affecting tomatoes and potatoes. It presents as water-soaked lesions that turn brown and necrotic, often with a white, fuzzy growth on the undersides. Rapid spread in humid conditions can devastate crops. Management includes removing infected plants, practicing good spacing, and promoting airflow. Fungicides are available for control, ideally applied preventively. Resistant varieties and early planting can help. Integrated strategies, such as cultural practices and monitoring, play a vital role in preventing and managing late blight. Consulting local agricultural resources is advised for effective solutions."),
    1: ("Tomato Yellow Leaf Curl Virus", "The Tomato Yellow Leaf Curl Virus (TYLCV) is a viral disease that affects tomato plants. It is transmitted by whiteflies and causes distinctive symptoms, including curling and yellowing of leaves, stunted growth, and reduced fruit yield. The virus interferes with the plant's ability to photosynthesize and develop properly. Management strategies include using insecticides to control whiteflies, planting resistant tomato varieties, and practicing good sanitation to reduce virus transmission. Early detection and removal of infected plants can also help prevent the spread of TYLCV. Integrated pest management and cultural practices are important for minimizing the impact of this virus on tomato crops."),
    2: ("Septoria leaf spot", "Septoria leaf spot is a fungal disease that commonly affects tomato plants. It displays as small circular spots with dark centers and lighter edges on leaves, leading to yellowing and leaf loss. The fungus spreads through water splashes, favoring wet conditions. Prevention methods include crop rotation, proper spacing, mulching, controlled watering, fungicides, pruning, using resistant varieties, and maintaining sanitation by removing infected debris after the season. These practices collectively help manage the disease and maintain tomato plant health."),
    3: ("Early blight", "Early blight is a fungal disease that affects tomato plants, causing dark concentric rings on leaves. As the disease progresses, leaves turn yellow, wither, and drop prematurely. It can also affect stems and fruit. The fungus thrives in warm, humid conditions. To manage early blight, practices like crop rotation, adequate spacing, mulching, controlled watering (avoiding overhead), proper pruning, and removing infected plant debris can help reduce its impact. Fungicides and planting resistant cultivars are also options for controlling early blight and ensuring healthy tomato growth."),
    4: ("spottedspider mite", "Spider mites, specifically the Two-Spotted Spider Mite, are tiny arachnids that can be damaging to tomato plants. These pests feed on plant sap by piercing the plant cells and sucking out their contents, leading to stippled, yellowing leaves and fine webbing on the undersides of leaves. The damage caused by spider mites can weaken the plants and reduce fruit quality. To manage spider mite infestations, regular monitoring of plants is essential to detect early signs of infestation. Practices such as maintaining proper plant hygiene, removing infested leaves, and creating a less favorable environment by increasing humidity and reducing drought stress can help deter spider mites. Additionally, natural predators like ladybugs and predatory mites can be introduced to control their population. If infestations are severe, using insecticidal soaps, neem oil, or horticultural oils can provide effective control."),
    5: ("Powdery mildew", "Powdery mildew is a common fungal disease that affects a wide range of plants, including tomatoes. It is caused by various species of the fungi belonging to the order Erysiphales. The disease is characterized by the appearance of white, powdery spots on the leaves, stems, and other plant parts. These spots are actually fungal growth and spores.The powdery mildew fungus thrives in conditions of high humidity and moderate temperatures, making it a common problem in climates with mild, damp weather. It can spread rapidly and impact the health and vigor of tomato plants."),
    6: ("Targat Spot", "Tomato Target Spot is a fungal disease that affects tomato plants, causing circular lesions with dark centers and light halos on leaves. Yellowing and premature defoliation can occur. Managing it involves cultural practices like sanitation, spacing, and mulching. Fungicides can be used preventively, and resistant tomato varieties can be chosen. Crop rotation, pruning, and organic treatments like neem oil are options. Integrated Pest Management (IPM) combines strategies for effective control. Consulting experts and local resources is crucial for tailored solutions."),
    7: ("Healthy", "Great ! This is a healthy leaf.."),
    8: ("Bacterial spot", "Bacterial spot is a common disease that affects tomato plants. It is caused by the bacterium Xanthomonas campestris pv. vesicatoria and results in dark water-soaked lesions on leaves, stems, and fruit. The disease can lead to reduced crop quality and yield. Preventive measures include using resistant varieties, practicing good sanitation, and employing proper irrigation techniques."),
    9: ("Tomato mosaic virus ", "Tomato mosaic virus (ToMV) is a plant pathogenic virus that infects tomatoes and other members of the Solanaceae family. It causes a mosaic pattern of light and dark green patches on the leaves, as well as stunted growth and reduced fruit quality. The virus is primarily spread through contaminated tools, seeds, and by aphids. Management strategies include using disease-free seeds, controlling aphid populations, and practicing crop rotation."),
    10: ("Leaf Mold", "Leaf mold, caused by the fungal pathogen *Fulvia fulva* (formerly *Cladosporium fulvum*), is a common disease in tomato plants. It typically appears as yellow or brown spots on the upper surface of leaves and a fuzzy white or gray growth on the undersides. The disease thrives in humid conditions and can spread rapidly through splashing water or contact. Preventing leaf mold involves providing proper spacing between plants to improve air circulation, avoiding overhead irrigation, and removing infected leaves. Fungicides can also be used for management, but integrated approaches that focus on cultural practices and resistant tomato varieties are more sustainable solutions."),
}


model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, in_channels=3)

model_path = 'efficientnet_b0_epoch_59.pth'
custom_classifier_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

model_dict = model.state_dict()
for key in custom_classifier_state_dict.keys():
    if key in model_dict:
        model_dict[key] = custom_classifier_state_dict[key]

model.load_state_dict(model_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_label(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    confidence = torch.max(output).item() * 100
    return predicted_class, confidence

@app.route("/", methods=['GET', 'POST'])
def main():
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    if request.method == 'POST':
        img = request.files["my_image"]
        img_path = 'static/' + img.filename
        img.save(img_path)
        predicted_class, confidence = predict_label(img_path)
        predicted_img_path = img.filename  # Use the filename as the predicted image path
        class_name, class_description = class_info.get(predicted_class, ("Unknown", "Description not available."))
        return render_template("index.html", predicted_class=class_name, confidence=confidence, predicted_img_path=predicted_img_path, current_time=current_time, class_description=class_description)
    return render_template("index.html", current_time=current_time)


if __name__ == '__main__':
    app.run(debug=True)
