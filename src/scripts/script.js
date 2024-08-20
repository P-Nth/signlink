document.getElementById('uploadForm').addEventListener('submit', async function (event) {
  // Prevent the default form submission behavior
  event.preventDefault();

  // Get the file input element
  const fileInput = document.getElementById('fileInput');

  // Retrieve the selected file from the file input
  const file = fileInput.files[0];

  // Check if a file was selected
  if (file) {
    // Create a FormData object to hold the file
    const formData = new FormData();
    formData.append('image', file);

    // Send a POST request to the server with the image file
    const response = await fetch('http://localhost:3000/predict', {
      method: 'POST',
      body: formData
    });

    // Parse the JSON response from the server
    const result = await response.json();

    // Display the predicted class in an HTML element with id 'translationText'
    document.getElementById('translationText').innerText = result.predicted_class;
  } else {
    // Alert the user if no file was selected
    alert('Please upload an image.');
  }
});

