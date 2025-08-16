# ParkingVision - AI-Powered Parking Management System

A modern parking management system with AI-powered video analysis and a React frontend.

## Project Structure

```
ParkingVision/
├── backend/                 # Python Flask API
│   ├── app.py              # Main Flask application
│   ├── core/               # Core parking monitoring logic
│   │   └── parking_monitor.py
│   ├── assets/             # Video files and coordinate data
│   ├── requirements.txt    # Python dependencies
│   └── Procfile           # Render deployment config
└── frontend/               # React frontend
    ├── src/
    │   ├── components/     # React components
    │   ├── pages/         # Page components
    │   ├── styles/        # CSS files
    │   ├── App.jsx        # Main app component
    │   └── main.jsx       # Entry point
    ├── package.json       # Node.js dependencies
    ├── vite.config.js     # Vite configuration
    ├── vercel.json        # Vercel deployment config
    └── index.html         # HTML template
```

## Features

- **Real-time Parking Monitoring**: AI-powered video analysis to detect parking spot occupancy
- **Modern React Frontend**: Responsive UI with real-time updates
- **Live Video Streaming**: Real-time video feeds from parking lots
- **Historical Data**: Track parking patterns over time
- **Mobile Responsive**: Works on all device sizes

## Technology Stack

### Backend
- **Python Flask**: RESTful API server
- **OpenCV**: Computer vision for parking spot detection
- **NumPy**: Numerical computations
- **Gunicorn**: Production WSGI server

### Frontend
- **React 18**: Modern UI framework
- **Vite**: Fast build tool and dev server
- **React Router**: Client-side routing
- **Axios**: HTTP client for API calls
- **CSS3**: Modern styling with CSS variables

## Local Development

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd ParkingVision/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```

The backend will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ParkingVision/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`

## Deployment

### Backend (Render)

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Set the following:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Environment**: Python 3

### Frontend (Vercel)

1. Push your code to GitHub
2. Import your repository on Vercel
3. Set the framework preset to "Vite"
4. Update the `vercel.json` file with your backend URL
5. Deploy

## API Endpoints

- `GET /api/parking-data` - Get current parking data for all lots
- `GET /api/parking-details?lot={lotId}` - Get detailed data for a specific lot
- `GET /api/video-stream?lot={lotId}` - Get live video stream for a lot

## Configuration

### Video Files
Place your parking lot video files in `backend/assets/`:
- `video-1.mp4` - Parking lot 1 video
- `video-3.mp4` - Parking lot 2 video

### Coordinate Files
Create coordinate files for parking spot positions:
- `coordinate-video-1` - Parking spots for lot 1
- `coordinate-video-3` - Parking spots for lot 2

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
