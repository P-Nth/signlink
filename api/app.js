const express = require('express');
const cors = require('cors');

const csv = require('csv-parser');
const multer = require('multer');
const path = require('path');
const ort = require('onnxruntime-node');
const winston = require('winston');
const Jimp = require('jimp');
const fs = require('fs');

const app = express();
const port = 3000;

/*
 * CORS Configuration
 *
 * Enables Cross-Origin Resource Sharing (CORS) to allow requests from
 * specific origins. In this case, it allows requests from
 * 'http://localhost:63342' with GET and POST methods and specific headers.
 */
const configureCors = () => {
  app.use(cors({
    origin: 'http://localhost:63342',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type']
  }));
};

/*
 * ONNX Runtime Initialization
 *
 * Loads the ONNX model into the ONNX runtime for performing inference.
 */
const initializeOnnxModel = () => {
  return ort.InferenceSession.create('main/slm_checkpoint.onnx');
};

/*
 * Multer Configuration
 *
 * Configures Multer to handle file uploads. Files are stored in memory
 * for processing.
 */
const configureMulter = () => {
  const storage = multer.memoryStorage();
  return multer({ storage: storage });
};

/*
 * Load Class Labels from CSV
 *
 * Reads and parses the CSV file containing class labels. Unique labels
 * are stored in an array for later use in mapping prediction indices to
 * class names.
 */
const loadClassLabels = (filePath) => {
  return new Promise((resolve, reject) => {
    const classLabels = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => {
        if (row.Label && !classLabels.includes(row.Label)) {
          classLabels.push(row.Label);
        }
      })
      .on('end', () => resolve(classLabels))
      .on('error', reject);
  });
};

/*
 * Resize Image
 *
 * Resizes the image to the target size (224x224) to match the input
 * shape of the model.
 */
const resizeImage = (image, targetSize = { width: 224, height: 224 }) => {
  return image.resize(targetSize.width, targetSize.height);
};

/*
 * Normalize Image
 *
 * Normalizes the pixel values of the image to be in the range [0, 1].
 */
const normalizeImage = (imageArray) => {
  return imageArray.map(pixel => pixel / 255.0);
};

/*
 * Image Preprocessing
 *
 * Processes the uploaded image to match the input requirements of the
 * ONNX model. This includes resizing, converting to grayscale, and
 * normalizing pixel values.
 */
const preprocessImage = (imageBuffer) => {
  return Jimp.read(imageBuffer)
    .then(image => resizeImage(image))
    .then(image => {
      image.grayscale();
      const imageArray = new Float32Array(224 * 224 * 3);

      image.scan(0, 0, 224, 224, (x, y, idx) => {
        const pixel = image.bitmap.data.readUInt8(idx);
        imageArray[y * 224 + x] = pixel;
      });

      const normalizedArray = normalizeImage(imageArray);
      return new ort.Tensor('float32', normalizedArray, [1, 224, 224, 3]); // Add batch dimension
    });
};

/*
 * Logger Configuration
 *
 * Sets up the Winston logger to handle logging of server activities.
 * Logs are output to both the console and a file named 'combined.log'.
 */
const createLogger = () => {
  return winston.createLogger({
    level: 'info',
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.json()
    ),
    transports: [
      new winston.transports.Console(),
      new winston.transports.File({ filename: 'combined.log' })
    ],
  });
};

/*
 * Prediction Endpoint Handler
 *
 * Handles POST requests to the '/predict' endpoint. It processes the
 * uploaded image, runs it through the ONNX model, and returns the
 * predicted class and class index. Logs various stages of processing
 * and any errors encountered.
 */
const handlePrediction = (session, classLabels, logger) => {
  return (req, res) => {
    logger.info('Received request for prediction');

    preprocessImage(req.file.buffer)
      .then(processedImage => {
        logger.info('Image processed');

        const feeds = { input: processedImage };
        return session.run(feeds);
      })
      .then(results => {
        const outputTensor = results['dense_1'];
        if (!outputTensor) {
          throw new Error('Model did not return an output tensor');
        }

        const predictionArray = outputTensor.data;
        const predictedClassIndex = predictionArray.indexOf(Math.max(...predictionArray));
        logger.info('Predicted class index:', { predictedClassIndex });

        const predictedClass = classLabels[predictedClassIndex] || 'Unknown';
        logger.info('Predicted class:', { predictedClass });

        res.json({
          predicted_class: predictedClass,
          predicted_class_index: predictedClassIndex
        });
      })
      .catch(error => {
        logger.error('Error during prediction:', { error: error.message });
        res.status(500).json({ error: 'An error occurred during prediction', details: error.message });
      });
  };
};

/*
 * Initialize and Start Server
 *
 * Initializes all necessary components and starts the Express server
 * listening on the specified port.
 */
const startServer = () => {
  configureCors();

  initializeOnnxModel()
    .then(session => loadClassLabels(path.join(__dirname, 'main', 'pred.csv'))
      .then(classLabels => {
        const logger = createLogger();
        const upload = configureMulter();

        app.post('/predict', upload.single('image'), handlePrediction(session, classLabels, logger));

        app.listen(port, () => {
          console.log(`Server running at http://localhost:${port}`);
        });
      })
    )
    .catch(error => {
      console.error('Error during server startup:', error);
    });
};

startServer();
