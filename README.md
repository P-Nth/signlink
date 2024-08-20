# SignLink

SignLink is an image recognition server application that processes images to predict their class labels using an ONNX model.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Features
- Image uploading and processing
- ONNX model inference for class prediction
- CORS enabled for cross-origin requests
- Logging with Winston

## Installation

### Prerequisites
- [Node.js](https://nodejs.org/) installed on your machine
- [Git](https://git-scm.com/) for version control

### Steps
1. Clone the repository:
   ```bash
   git clone git@github.com:P-Nth/signlink.git

2. Install Dependencies:
   ```bash
   npm install

3. Create a .gitignore File:
   ```bash
   node_modules/
   logs/
   *.log
   .env
   .vscode/

4. Running the Server:
   ```bash
   node app.js

5. Making a Prediction
   ```bash
   curl -X POST http://localhost:3000/predict -F "image=@/path/to/your/image.jpg"
