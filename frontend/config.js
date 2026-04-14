// ==========================================
// DEPLOYMENT CONFIGURATION
// ==========================================

// This dynamically configures the backend connection based on where you host the frontend.
// For local development, it automatically connects via 'http://localhost:5000'
// For production, it directs traffic securely to your Render web instance.

let API_URL;

if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.protocol === "file:") {
    API_URL = 'http://localhost:5000';
} else {
    // NOTE: When deploying to GitHub Pages, Netlify, or Vercel, replace the string below
    // with the exact Web Service URL generated when you deploy the Docker container to Render.
    API_URL = 'https://indiproctor.onrender.com';
}
