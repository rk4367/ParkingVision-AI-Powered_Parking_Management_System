import React, { useState, useEffect } from 'react'
import ParkingCard from '../components/ParkingCard'
import axios from 'axios'
import './Dashboard.css'

const Dashboard = () => {
  const [parkingData, setParkingData] = useState({
    lot1: { total: 0, available: 0, occupied: 0 },
    lot2: { total: 0, available: 0, occupied: 0 }
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchParkingData = async () => {
    try {
      const response = await axios.get('/api/parking-data')
      setParkingData(response.data)
      setError(null)
    } catch (err) {
      console.error('Error fetching parking data:', err)
      setError('Failed to fetch parking data')
      // Use mock data for development
      setParkingData({
        lot1: { total: 0, available: 0, occupied: 0 },
        lot2: { total: 0, available: 0, occupied: 0 }
      })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchParkingData()
    const interval = setInterval(fetchParkingData, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading">Loading parking data...</div>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <h1>Parking Monitoring System</h1>
      
      {error && (
        <div className="error-message">
          {error} - Using demo data
        </div>
      )}
      
      <div className="cards-container">
        <ParkingCard lotId={1} data={parkingData.lot1} />
        <ParkingCard lotId={2} data={parkingData.lot2} />
      </div>
    </div>
  )
}

export default Dashboard 