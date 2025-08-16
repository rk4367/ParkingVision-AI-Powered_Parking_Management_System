import React from 'react'
import { useNavigate } from 'react-router-dom'
import './ParkingCard.css'

const ParkingCard = ({ lotId, data }) => {
  const navigate = useNavigate()

  const handleMoreInfo = () => {
    navigate(`/details/${lotId}`)
  }

  return (
    <div className="parking-card">
      <h2>Parking Lot {lotId}</h2>
      <div className="stat-container">
        <div className="stat">
          <div className="stat-value">{data.total || 0}</div>
          <div className="stat-label">Total Slots</div>
        </div>
        <div className="stat">
          <div className="stat-value available">{data.available || 0}</div>
          <div className="stat-label">Available</div>
        </div>
        <div className="stat">
          <div className="stat-value occupied">{data.occupied || 0}</div>
          <div className="stat-label">Occupied</div>
        </div>
      </div>
      <button className="more-info-btn" onClick={handleMoreInfo}>
        More Info
      </button>
    </div>
  )
}

export default ParkingCard 