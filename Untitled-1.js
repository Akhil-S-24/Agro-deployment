// App.js
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import axios from 'axios';

// Components
const Register = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    role: 'farmer',
    name: '',
    email: '',
    phone: '',
    address: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/register', formData);
      alert(response.data.message);
    } catch (error) {
      alert(error.response.data.error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Register</h2>
      <select name="role" value={formData.role} onChange={(e) => setFormData({...formData, role: e.target.value})}>
        <option value="farmer">Farmer</option>
        <option value="customer">Customer</option>
      </select>
      <input type="text" placeholder="Username" value={formData.username} onChange={(e) => setFormData({...formData, username: e.target.value})} />
      <input type="password" placeholder="Password" value={formData.password} onChange={(e) => setFormData({...formData, password: e.target.value})} />
      <input type="text" placeholder="Full Name" value={formData.name} onChange={(e) => setFormData({...formData, name: e.target.value})} />
      <input type="email" placeholder="Email" value={formData.email} onChange={(e) => setFormData({...formData, email: e.target.value})} />
      <input type="tel" placeholder="Phone" value={formData.phone} onChange={(e) => setFormData({...formData, phone: e.target.value})} />
      <textarea placeholder="Address" value={formData.address} onChange={(e) => setFormData({...formData, address: e.target.value})} />
      <button type="submit">Register</button>
    </form>
  );
};

const Login = () => {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [user, setUser] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/login', credentials);
      setUser(response.data.user);
      alert('Login successful');
    } catch (error) {
      alert('Login failed');
    }
  };

  return (
    <div>
      {!user ? (
        <form onSubmit={handleSubmit}>
          <h2>Login</h2>
          <input type="text" placeholder="Username" value={credentials.username} onChange={(e) => setCredentials({...credentials, username: e.target.value})} />
          <input type="password" placeholder="Password" value={credentials.password} onChange={(e) => setCredentials({...credentials, password: e.target.value})} />
          <button type="submit">Login</button>
        </form>
      ) : (
        <div>
          <h3>Welcome, {user.name}</h3>
          <p>Role: {user.role}</p>
        </div>
      )}
    </div>
  );
};

const CropRecommendation = () => {
  const [formData, setFormData] = useState({
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: ''
  });
  const [recommendation, setRecommendation] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/recommend-crop', formData);
      setRecommendation(response.data);
    } catch (error) {
      alert('Error getting recommendation');
    }
  };

  return (
    <div>
      <h2>Crop Recommendation</h2>
      <form onSubmit={handleSubmit}>
        <input type="number" placeholder="Nitrogen" value={formData.nitrogen} onChange={(e) => setFormData({...formData, nitrogen: e.target.value})} />
        <input type="number" placeholder="Phosphorus" value={formData.phosphorus} onChange={(e) => setFormData({...formData, phosphorus: e.target.value})} />
        <input type="number" placeholder="Potassium" value={formData.potassium} onChange={(e) => setFormData({...formData, potassium: e.target.value})} />
        <input type="number" placeholder="Temperature" value={formData.temperature} onChange={(e) => setFormData({...formData, temperature: e.target.value})} />
        <input type="number" placeholder="Humidity" value={formData.humidity} onChange={(e) => setFormData({...formData, humidity: e.target.value})} />
        <input type="number" placeholder="pH" value={formData.ph} onChange={(e) => setFormData({...formData, ph: e.target.value})} step="0.1" />
        <input type="number" placeholder="Rainfall" value={formData.rainfall} onChange={(e) => setFormData({...formData, rainfall: e.target.value})} />
        <button type="submit">Get Recommendation</button>
      </form>
      
      {recommendation && (
        <div>
          <h3>Recommended Crop: {recommendation.recommended_crop}</h3>
          <p>Confidence: {(recommendation.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
};

const Marketplace = () => {
  const [crops, setCrops] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchCrops = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/crops');
      setCrops(response.data);
    } catch (error) {
      alert('Error fetching crops');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    fetchCrops();
  }, []);

  return (
    <div>
      <h2>Crop Marketplace</h2>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div>
          {crops.map(crop => (
            <div key={crop.id} style={{border: '1px solid #ccc', padding: '10px', margin: '10px'}}>
              <h3>{crop.name}</h3>
              <p>{crop.description}</p>
              <p>Price: ${crop.price} per kg</p>
              <p>Available: {crop.quantity} kg</p>
              <p>Sold by: {crop.farmer_name}</p>
              <button>Add to Cart</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const App = () => {
  return (
    <Router>
      <nav>
        <Link to="/">Home</Link> | 
        <Link to="/register">Register</Link> | 
        <Link to="/login">Login</Link> | 
        <Link to="/crop-recommendation">Crop Recommendation</Link> | 
        <Link to="/marketplace">Marketplace</Link>
      </nav>
      
      <Routes>
        <Route path="/" element={<h1>Agricultural Management System</h1>} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/crop-recommendation" element={<CropRecommendation />} />
        <Route path="/marketplace" element={<Marketplace />} />
      </Routes>
    </Router>
  );
};

export default App;