# Dava Drishti (दाव दृष्टि)
"Fire Vision" - Combining "Dāva" (forest fire) with "Drishti" (vision/sight)

A comprehensive AI/ML-powered forest fire prediction and simulation system that generates next-day fire probability maps and simulates fire spread dynamics for forest management and emergency response applications.

## 🎯 Project Overview

This system combines advanced deep learning architectures (U-Net, LSTM) with cellular automata modeling to provide:
- **High-resolution fire probability maps** (30m pixel resolution)
- **Dynamic fire spread simulations** (1, 2, 3, 6, and 12-hour horizons)
- **Real-time web-based platform** for visualization and analysis
- **Production-ready API** for integration with external systems

## 🏗️ System Architecture

### Core Components
- **Data Pipeline**: Automated collection and preprocessing of multi-source geospatial data
- **AI/ML Models**: U-Net for spatial prediction, LSTM for temporal patterns
- **Cellular Automata**: Dynamic fire spread simulation engine
- **Web Application**: Interactive mapping and visualization platform
- **API Services**: RESTful endpoints for model serving

### Technology Stack
- **Backend**: Python 3.9+, Flask/FastAPI, TensorFlow/Keras
- **Frontend**: HTML5/CSS3/JavaScript, Leaflet.js, Chart.js
- **Database**: PostgreSQL with PostGIS, MongoDB
- **Deployment**: Docker, Docker Compose, CI/CD pipeline

## 📊 Data Sources

### Weather Data
- ERA5 reanalysis data (meteorological variables)
- MOSDAC/IMD (localized Indian weather data)
- Variables: temperature, humidity, wind speed/direction, precipitation

### Terrain & Topographic Data
- 30m resolution DEM from Bhoonidhi portal
- Derived parameters: slope, aspect, elevation

### Vegetation & Land Cover
- LULC datasets from Bhuvan/Sentinel Hub
- NDVI and vegetation indices
- Fuel load estimation algorithms

### Fire Detection Data
- VIIRS active fire data from NASA FIRMS
- Historical fire occurrence records

### Infrastructure Data
- GHSL (Global Human Settlement Layer)
- Road networks and settlement proximity

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- PostgreSQL with PostGIS extension
- MongoDB

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Vincenzo-Verma/Dava-Drishti.git
cd bah_v1_cu
```

2. **Set up environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start services with Docker**
```bash
docker-compose up -d
```

5. **Initialize database**
```bash
python scripts/init_database.py
```

6. **Run the application**
```bash
python app.py
```

The web application will be available at `http://localhost:5000`

## 📁 Project Structure

```
bah_v1_cu/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── models/                   # ML model implementations
│   │   ├── unet_model.py        # U-Net architecture
│   │   ├── lstm_model.py        # LSTM temporal model
│   │   └── cellular_automata.py # Fire spread simulation
│   ├── data/                    # Data processing modules
│   │   ├── collectors/          # Data collection scripts
│   │   ├── processors/          # Data preprocessing
│   │   └── validators/          # Data quality validation
│   ├── api/                     # REST API endpoints
│   ├── web/                     # Web interface
│   └── utils/                   # Utility functions
├── config/                      # Configuration files
├── data/                        # Data storage
│   ├── raw/                     # Raw data downloads
│   ├── processed/               # Processed datasets
│   └── models/                  # Trained model files
├── scripts/                     # Utility scripts
├── tests/                       # Test suite
├── docs/                        # Documentation
├── docker/                      # Docker configurations
├── requirements.txt             # Python dependencies
├── docker-compose.yml          # Service orchestration
└── README.md                   # This file
```

## 🎯 Performance Targets

### Model Performance
- **Accuracy**: ≥90% overall classification accuracy
- **F1-Score**: ≥0.85 balanced performance
- **Spatial Accuracy**: Fire boundary prediction within 95% confidence intervals
- **Temporal Accuracy**: Fire spread timing predictions within ±10% of observed values

### System Performance
- **API Response Time**: <2 seconds for prediction requests
- **System Uptime**: >99.5% availability
- **Data Processing**: >95% pipeline completion rate

## 🔧 Development Phases

### Phase 1: Data Pipeline Development (Weeks 1-3)
- [x] Automated data collection scripts
- [x] Data preprocessing pipelines
- [x] Quality control frameworks
- [x] 30m resolution feature stack creation

### Phase 2: Model Development (Weeks 4-8)
- [ ] U-Net model training
- [ ] LSTM temporal model
- [ ] Cellular automata simulation
- [ ] Model validation and evaluation

### Phase 3: Web Application (Weeks 9-11)
- [ ] Interactive web interface
- [ ] RESTful API endpoints
- [ ] Real-time visualisation
- [ ] User authentication

### Phase 4: Testing & Deployment (Weeks 12-14)
- [ ] Comprehensive test suite
- [ ] Docker deployment
- [ ] Monitoring systems
- [ ] Documentation

## 📈 Usage Examples

### Generate Fire Risk Map
```python
from app.models import FirePredictionModel

# Initialise model
model = FirePredictionModel()

# Generate next-day fire risk map
risk_map = model.predict_fire_risk(
    region="forest_area_1",
    date="2024-01-15"
)

# Save as GeoTIFF
risk_map.save("fire_risk_2024_01_15.tif")
```

### Run Fire Spread Simulation
```python
from app.models import FireSpreadSimulation

# Initialise simulation
sim = FireSpreadSimulation()

# Run 6-hour fire spread simulation
spread_animation = sim.simulate_fire_spread(
    ignition_point=(lat, lon),
    duration_hours=6,
    wind_conditions=current_wind_data
)

# Generate animation
spread_animation.save("fire_spread_6h.gif")
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html

# Run specific test categories
python -m pytest tests/test_models.py
python -m pytest tests/test_api.py
```

## 📚 API Documentation

### Fire Prediction Endpoints

#### GET /api/v1/fire-risk/{region}
Get fire risk prediction for a specific region.

**Parameters:**
- `region` (string): Region identifier
- `date` (string, optional): Date in YYYY-MM-DD format

**Response:**
```json
{
  "region": "forest_area_1",
  "date": "2024-01-15",
  "risk_map_url": "/api/v1/maps/fire_risk_2024_01_15.tif",
  "risk_level": "high",
  "confidence": 0.92
}
```

#### POST /api/v1/fire-simulation
Run fire spread simulation.

**Request Body:**
```json
{
  "ignition_point": {"lat": 12.9716, "lon": 77.5946},
  "duration_hours": 6,
  "wind_conditions": {
    "speed": 15.5,
    "direction": 270
  }
}
```

**Response:**
```json
{
  "simulation_id": "sim_12345",
  "animation_url": "/api/v1/animations/sim_12345.gif",
  "affected_area_km2": 45.2,
  "completion_time": "2024-01-15T10:30:00Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the [documentation](docs/) for detailed guides

## 🔮 Roadmap

- [ ] Integration with satellite imagery for real-time monitoring
- [ ] Machine learning model retraining pipeline
- [ ] Mobile application for field personnel
- [ ] Integration with emergency response systems
- [ ] Advanced visualisation features (3D terrain, VR support)
- [ ] Multi-language support for international deployment

---

**Note**: This system is designed for research and operational use in forest fire management. Always validate predictions with ground observations and follow local fire management protocols. 
