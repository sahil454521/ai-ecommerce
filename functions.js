async function predictPrice() {
    const productName = document.getElementById('productName').value.trim();
    
    if (!productName) {
        alert("Please enter a product name!");
        return;
    }

    try {
        const response = await fetch('http://localhost:5000/predict-price', {  // Add the full URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ product_name: productName }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Add result display function
        const resultDiv = document.getElementById('result');
        if (data.status === 'success') {
            resultDiv.innerHTML = `
                <div class="result-box">
                    <h3>Price Prediction for ${data.product}</h3>
                    <p>Current Price: $${data.current_price}</p>
                    <p>Predicted Price: $${data.predicted_price}</p>
                    <p>Trend: ${data.trend}</p>
                </div>
            `;
        } else {
            showError(data.error || "Failed to predict price");
        }
    } catch (error) {
        console.error("Error:", error);
        showError("An unexpected error occurred. Please try again.");
    }
}

// Notification helper
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
      notification.classList.add('fade-out');
      setTimeout(() => notification.remove(), 500);
  }, 3000);
}

// Add product selection functionality
document.querySelectorAll('.product-cart').forEach(product => {
  product.addEventListener('click', function() {
      document.querySelectorAll('.product-cart').forEach(p => p.classList.remove('selected-product'));
      this.classList.add('selected-product');
      const productName = this.querySelector('h4:not(.price)').textContent.trim();
      predictPrice(productName);
  });
});
async function fetchLeaderboard() {
    try {
        const response = await fetch('http://localhost:5000/leaderboard');
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        const resultDiv = document.getElementById('result');
        
        if (data.leaderboard && data.leaderboard.length > 0) {
            resultDiv.innerHTML = `
                <div class="result-box">
                    <h3>Seller Leaderboard</h3>
                    <ul>
                        ${data.leaderboard.map(seller => `
                            <li>${seller.seller}: $${seller.total_sales}</li>
                        `).join('')}
                    </ul>
                </div>
            `;
        } else {
            showError("No leaderboard data available.");
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to fetch leaderboard: ' + error.message);
    }
}