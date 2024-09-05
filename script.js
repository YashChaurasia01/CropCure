let model;
const classes = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
};

async function loadModel() {
    try {
        model = await tf.loadGraphModel('./tfjs_model/model.json');
        const modeltag = document.getElementById('modelTag');
        modeltag.innerHTML = `<u>Use Model( Loaded! )</u>`;
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
    }
}

function handleImageUpload() {
    const imageInput = document.getElementById('imageInput');
    const imageContainer = document.getElementById('imageContainer');
    const file = imageInput.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            const img = new Image();
            img.onload = function () {
                // Display the selected image
                imageContainer.innerHTML = '';
                imageContainer.appendChild(img);
            };

            img.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }
}

function detectImage() {
    if (!model) {
        console.error('Model not loaded yet');
        return;
    }
    const imageInput = document.getElementById('imageInput');
    const imageContainer = document.getElementById('imageContainer');
    const file = imageInput.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = async function (e) {
            const img = new Image();
            img.onload = async function () {

                // Convert the resized image to a TensorFlow tensor for prediction
                const imageTensor = tf.browser.fromPixels(img)
                    .resizeNearestNeighbor([256, 256])
                    .toFloat()
                    .expandDims();
                const scaledImg = imageTensor.div(255);

                // Make predictions on the scaled image
                const prediction = await model.predict(scaledImg).data();
                const predictedClass = classes[prediction.indexOf(Math.max(...prediction))];

                imageContainer.innerHTML += `<p class='mt-2 fw-bold fs-1'>Predicted Class: ${predictedClass}</p>`;
            };

            img.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }
}

// Load the model when the page is loaded
window.onload = function () {
    loadModel();
    // Call the handleImageUpload function when an image is chosen
    document.getElementById('imageInput').addEventListener('change', handleImageUpload);
};
