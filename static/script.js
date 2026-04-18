document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const predictedCropElement = document.getElementById('predicted-crop');
    const modelAccuracyElement = document.getElementById('model-accuracy');
    const errorMessageElement = document.getElementById('error-message');

    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent page reload

        // Hide previous results/errors
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        
        // Update button state
        const originalBtnText = submitBtn.innerText;
        submitBtn.innerText = 'Predicting...';
        submitBtn.disabled = true;

        // Gather form data
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }

        try {
            // Send data to backend using fetch API (AJAX)
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (response.ok) {
                // Show result dynamically
                predictedCropElement.innerText = result.prediction;
                modelAccuracyElement.innerText = `Model Accuracy: ${result.accuracy}`;
                resultContainer.classList.remove('hidden');
            } else {
                // Show error if backend returns an error
                throw new Error(result.error || 'Something went wrong');
            }
        } catch (error) {
            // Display error message
            errorMessageElement.innerText = error.message;
            errorContainer.classList.remove('hidden');
        } finally {
            // Restore button state
            submitBtn.innerText = originalBtnText;
            submitBtn.disabled = false;
        }
    });
});
