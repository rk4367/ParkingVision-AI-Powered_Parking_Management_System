import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import ParkingDetails from './pages/ParkingDetails'
import Footer from './components/Footer'
import './styles/App.css'

function App() {
  return (
    <div className="App">
      <Navbar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/details/:lotId" element={<ParkingDetails />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App 