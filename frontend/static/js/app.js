const API_BASE_URL = 'http://localhost:5000/api';
let systemInitialized = false;

async function initializeSystem() {
    updateStatus('Initializing system...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/initialize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        systemInitialized = true;
        updateStatus('System initialized successfully', 'success');
        updateSystemStatus('Online');
        
        console.log('Initialization result:', data);
        
        showAlert('System initialized successfully! Models trained and services loaded.', 'success');
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Initialization failed', 'error');
        showAlert(`Initialization failed: ${error.message}`, 'error');
    }
}

async function searchParking() {
    if (!systemInitialized) {
        showAlert('Please initialize the system first', 'warning');
        return;
    }
    
    const lat = parseFloat(document.getElementById('searchLat').value);
    const lon = parseFloat(document.getElementById('searchLon').value);
    const distance = parseFloat(document.getElementById('searchDistance').value);
    const rate = parseFloat(document.getElementById('searchRate').value);
    
    updateStatus('Searching for parking spots...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/parking/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                latitude: lat,
                longitude: lon,
                max_distance: distance,
                max_rate: rate,
                top_n: 10
            })
        });
        
        const data = await response.json();
        
        displaySearchResults(data.spots);
        updateStatus(`Found ${data.total_results} parking spots`, 'success');
    } catch (error) {
        console.error('Search error:', error);
        showAlert(`Search failed: ${error.message}`, 'error');
    }
}

function displaySearchResults(spots) {
    const resultsContainer = document.getElementById('searchResults');
    
    if (!spots || spots.length === 0) {
        resultsContainer.innerHTML = '<p>No parking spots found matching your criteria.</p>';
        return;
    }
    
    let html = '';
    
    spots.forEach((spot, index) => {
        html += `
            <div class="result-item">
                <h4>Spot ${index + 1}: ${spot.spot_id}</h4>
                <p><strong>Type:</strong> ${spot.spot_type}</p>
                <p><strong>Zone:</strong> ${spot.zone}</p>
                <p><strong>Distance:</strong> ${spot.distance_km.toFixed(2)} km</p>
                <p><strong>Rate:</strong> $${spot.hourly_rate.toFixed(2)}/hr</p>
                <p><strong>Availability Score:</strong> <span class="score">${(spot.availability_score * 100).toFixed(1)}%</span></p>
                <p><strong>Composite Score:</strong> <span class="score">${(spot.composite_score * 100).toFixed(1)}%</span></p>
                <p><strong>Location:</strong> ${spot.latitude.toFixed(6)}, ${spot.longitude.toFixed(6)}</p>
            </div>
        `;
    });
    
    resultsContainer.innerHTML = html;
}

async function getRecommendations() {
    if (!systemInitialized) {
        showAlert('Please initialize the system first', 'warning');
        return;
    }
    
    const duration = parseFloat(document.getElementById('recDuration').value);
    const accessibility = document.getElementById('recAccessibility').checked;
    
    const lat = parseFloat(document.getElementById('searchLat').value);
    const lon = parseFloat(document.getElementById('searchLon').value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/parking/recommendations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                latitude: lat,
                longitude: lon,
                duration_hours: duration,
                preferences: {
                    accessibility: accessibility,
                    max_distance: 2.0
                }
            })
        });
        
        const data = await response.json();
        
        displayRecommendations(data.recommendations);
        updateStatus(`Generated ${data.recommendations.length} recommendations`, 'success');
    } catch (error) {
        console.error('Recommendations error:', error);
        showAlert(`Failed to get recommendations: ${error.message}`, 'error');
    }
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p>No recommendations available.</p>';
        return;
    }
    
    let html = '';
    
    recommendations.forEach((rec, index) => {
        const confidenceClass = `confidence-${rec.confidence}`;
        
        html += `
            <div class="recommendation-card">
                <h4>Recommendation ${index + 1}</h4>
                <div class="detail">
                    <span>Spot ID:</span>
                    <span>${rec.spot_id}</span>
                </div>
                <div class="detail">
                    <span>Zone:</span>
                    <span>${rec.zone}</span>
                </div>
                <div class="detail">
                    <span>Distance:</span>
                    <span>${rec.distance_km.toFixed(2)} km (${rec.walking_time_minutes} min walk)</span>
                </div>
                <div class="detail">
                    <span>Estimated Cost:</span>
                    <span>$${rec.estimated_cost.toFixed(2)}</span>
                </div>
                <div class="detail">
                    <span>Availability:</span>
                    <span>${(rec.availability_probability * 100).toFixed(1)}%</span>
                </div>
                <div class="detail">
                    <span>Confidence:</span>
                    <span class="${confidenceClass}">${rec.confidence.toUpperCase()}</span>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

async function loadZoneStats() {
    if (!systemInitialized) {
        showAlert('Please initialize the system first', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/parking/zones`);
        const data = await response.json();
        
        displayZoneStatsChart(data.zones);
        displayZoneStatsTable(data.zones);
        
        updateStatus(`Loaded statistics for ${data.total_zones} zones`, 'success');
    } catch (error) {
        console.error('Zone stats error:', error);
        showAlert(`Failed to load zone statistics: ${error.message}`, 'error');
    }
}

function displayZoneStatsChart(zones) {
    const zoneNames = Object.keys(zones);
    const occupancyRates = zoneNames.map(zone => zones[zone].occupancy_rate * 100);
    const avgRates = zoneNames.map(zone => zones[zone].average_hourly_rate);
    
    const trace1 = {
        x: zoneNames,
        y: occupancyRates,
        type: 'bar',
        name: 'Occupancy Rate (%)',
        marker: { color: '#667eea' }
    };
    
    const trace2 = {
        x: zoneNames,
        y: avgRates,
        type: 'bar',
        name: 'Avg Rate ($/hr)',
        yaxis: 'y2',
        marker: { color: '#48bb78' }
    };
    
    const layout = {
        title: 'Zone Statistics Overview',
        xaxis: { title: 'Zone' },
        yaxis: { title: 'Occupancy Rate (%)' },
        yaxis2: {
            title: 'Average Rate ($/hr)',
            overlaying: 'y',
            side: 'right'
        },
        barmode: 'group',
        height: 400
    };
    
    Plotly.newPlot('zoneStatsChart', [trace1, trace2], layout);
}

function displayZoneStatsTable(zones) {
    const container = document.getElementById('zoneStatsTable');
    
    let html = '<table><thead><tr>';
    html += '<th>Zone</th><th>Total Spots</th><th>Available</th><th>Occupancy</th><th>Avg Rate</th>';
    html += '</tr></thead><tbody>';
    
    Object.keys(zones).forEach(zoneName => {
        const zone = zones[zoneName];
        html += '<tr>';
        html += `<td>${zoneName}</td>`;
        html += `<td>${zone.total_spots}</td>`;
        html += `<td>${zone.available_spots}</td>`;
        html += `<td>${(zone.occupancy_rate * 100).toFixed(1)}%</td>`;
        html += `<td>$${zone.average_hourly_rate.toFixed(2)}</td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    container.innerHTML = html;
}

async function loadPatterns() {
    if (!systemInitialized) {
        showAlert('Please initialize the system first', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/parking/patterns`);
        const data = await response.json();
        
        displayPeakHoursChart(data.peak_hours);
        displayWeekdayPatternChart(data.weekday_pattern, data.weekend_pattern);
        displayRevenueChart(data);
        
        updateStatus('Loaded historical patterns', 'success');
    } catch (error) {
        console.error('Patterns error:', error);
        showAlert(`Failed to load patterns: ${error.message}`, 'error');
    }
}

function displayPeakHoursChart(peakHours) {
    const hours = Object.keys(peakHours).map(h => parseInt(h));
    const rates = Object.values(peakHours);
    
    const trace = {
        x: hours,
        y: rates,
        type: 'bar',
        marker: { color: '#667eea' }
    };
    
    const layout = {
        title: 'Peak Hours Occupancy',
        xaxis: { title: 'Hour of Day' },
        yaxis: { title: 'Occupancy Rate' },
        height: 300
    };
    
    Plotly.newPlot('peakHoursChart', [trace], layout);
}

function displayWeekdayPatternChart(weekdayPattern, weekendPattern) {
    const hours = Object.keys(weekdayPattern).map(h => parseInt(h));
    
    const trace1 = {
        x: hours,
        y: Object.values(weekdayPattern),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Weekday',
        line: { color: '#667eea', width: 3 }
    };
    
    const trace2 = {
        x: hours,
        y: Object.values(weekendPattern),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Weekend',
        line: { color: '#48bb78', width: 3 }
    };
    
    const layout = {
        title: 'Weekday vs Weekend Patterns',
        xaxis: { title: 'Hour of Day' },
        yaxis: { title: 'Occupancy Rate' },
        height: 300
    };
    
    Plotly.newPlot('weekdayPatternChart', [trace1, trace2], layout);
}

function displayRevenueChart(data) {
    const trace = {
        values: [data.total_revenue, data.average_turnover_rate * 10000],
        labels: ['Total Revenue ($)', 'Turnover Rate (scaled)'],
        type: 'pie',
        marker: {
            colors: ['#667eea', '#48bb78']
        }
    };
    
    const layout = {
        title: 'Revenue Metrics',
        height: 300
    };
    
    Plotly.newPlot('revenueChart', [trace], layout);
}

async function getModelMetrics() {
    if (!systemInitialized) {
        showAlert('Please initialize the system first', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/models/metrics`);
        const data = await response.json();
        
        displayModelMetrics(data);
        updateStatus('Loaded model metrics', 'success');
    } catch (error) {
        console.error('Model metrics error:', error);
        showAlert(`Failed to load model metrics: ${error.message}`, 'error');
    }
}

function displayModelMetrics(metrics) {
    const container = document.getElementById('modelMetrics');
    
    let html = '<div class="metrics-container"><h4>Model Performance</h4>';
    
    Object.keys(metrics).forEach(modelName => {
        const model = metrics[modelName];
        html += '<div class="metric-row">';
        html += `<span><strong>${modelName.toUpperCase()}:</strong></span>`;
        html += `<span>${model.trained ? 'Trained' : 'Not Trained'} (${model.model_type || 'N/A'})</span>`;
        html += '</div>';
    });
    
    html += '</div>';
    
    container.innerHTML = html;
}

async function refreshData() {
    if (!systemInitialized) {
        showAlert('Please initialize the system first', 'warning');
        return;
    }
    
    updateStatus('Refreshing data...', 'info');
    
    await loadZoneStats();
    await loadPatterns();
    
    updateStatus('Data refreshed', 'success');
    updateLastUpdate();
}

function updateSystemStatus(status) {
    const statusElement = document.getElementById('systemStatus');
    if (statusElement) {
        statusElement.textContent = status;
        statusElement.style.color = status === 'Online' ? '#48bb78' : '#f56565';
    }
}

function updateLastUpdate() {
    const updateElement = document.getElementById('lastUpdate');
    if (updateElement) {
        const now = new Date();
        updateElement.textContent = now.toLocaleTimeString();
    }
}

function updateStatus(message, type) {
    console.log(`[${type.toUpperCase()}] ${message}`);
    updateLastUpdate();
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('Parking Finder Dashboard loaded');
    updateSystemStatus('Not Initialized');
    updateLastUpdate();
});