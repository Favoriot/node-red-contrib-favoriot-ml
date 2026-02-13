# node-red-contrib-favoriot-ml

A high-performance Node-RED node for executing ONNX machine learning models training in Favoriot Platform. This node bridges the gap between raw data and model inference by providing built-in support for categorical encoding and feature scaling.
---

## 🚀 Overview

The **Favoriot ML Inference** node simplifies the deployment of machine learning models within Node-RED. Unlike generic ONNX nodes, this package includes a sophisticated preprocessing engine that automatically handles string-to-numeric encoding and data normalization.


## 📦 Installation

To install this node directly from the GitHub source, follow these steps in your terminal:
1. Navigate to your Node-RED user directory (usually ~/.node-red):
    ```bash
      cd ~/.node-red
    ```
2. Install the package directly using the GitHub URL:
    ```bash
      npm install https://github.com/Favoriot/node-red-contrib-favoriot-ml
    ```
3. Restart your Node-RED instance to load the new node into the palette.

## ✨ Key Features
- Categorical Encoding: Pass raw strings (e.g., "High", "Visitor") directly. The node uses pre-configured maps to convert them to numeric values.
- Automatic Scaling: Built-in StandardScaler logic (mean/std) to normalize numeric inputs before they reach the model.
- Feature Alignment: Automatically reorders JSON input keys to match the specific tensor order required by your ONNX model.
- Robustness: Graceful error handling for unknown categories or missing features to prevent flow crashes.

## 📖 Usage
1. Input Format (msg.payload)
You can send mixed data types. The node handles the conversion for you
```json
{
  "User_Type": "visitor",
  "Traffic_Level": "high",
  "Temperature": 30.5,
  "Humidity": 82
}
```
2. Output Format
The node appends results to the message object:
```json
{
  "prediction": 1,
  "status": "success",
  "timestamp": "2026-02-13T12:00:00.000Z",
  "features": ["User_Type", "Traffic_Level", "Temperature", "Humidity"],
  "feature_values": ["visitor", "high", 30.5, 82]
}
```

## ⚙️ Requirements
- Node.js: v14.0.0 or higher
- Node-RED: v2.0.0 or higher
- Platforms: Windows, Linux, macOS (via onnxruntime-node)

## 📝 License
This project is licensed under the MIT License.

## 👤 Author
mnazrinnapiah - Initial work - Favoriot

## 🤝 Support

- For Favoriot-specific integration, visit the **Favoriot Developer Documentations**:  
  https://platform.favoriot.com/tutorial/v2/

- 🌐 Official Website:  
  https://favoriot.com/

- 💻 GitHub Repository:  
  https://github.com/Favoriot
