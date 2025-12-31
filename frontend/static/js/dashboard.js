const API_BASE_URL = 'http://localhost:5000/api';
let updateInterval = null;

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    loadDashboardData();
    startAutoRefresh();
});

async function loadDashboardData() {
    try {
        await Promise.all([
            updateMetrics(),
            loadRealtimeOccupancy(),
            loadTopZones(),
            loadPerformanceMetrics(),
            loadRecentActivity()
        ]);
        
        updateLastUpdateTime();
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

async function updateMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/parking/zones`);
        const data = await response.json();
        
        let totalSpots = 0;
        let availableSpots = 0;
        
        Object.values(data.zones).forEach(zone => {
            totalSpots += zone.total_spots;
            availableSpots += zone.available_spots;
        });
        
        const occupancyRate = ((totalSpots - availableSpots) / totalSpots * 100).toFixed(1);
        
        document.getElementById('totalSpots').textContent = totalSpots.toLocaleString();
        document.getElementById('availableSpots').textContent = availableSpots.toLocaleString();
        document.getElementById('occupancyRate').textContent = `${occupancyRate}%`;
        
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

function loadRealtimeOccupancy() {
    const hours = Array.from({length: 24}, (_, i) => i);
    const current = new Date().getHours();
    const occupancy = hours.map(h => {
        const distance = Math.abs(h - current);
        if (distance < 2) return 0.75 + Math.random() * 0.15;
        if (distance < 5) return 0.60 + Math.random() * 0.15;
        return 0.45 + Math.random() * 0.20;
    });
    
    const trace = {
        x: hours,
        y: occupancy,
        type: 'scatter',
        mode: 'lines+markers',
        fill: 'tozeroy',
        line: { color: '#667eea', width: 3 },
        marker: { size: 8 }
    };
    
    const layout = {
        xaxis: { title: 'Hour of Day' },
        yaxis: { title: 'Occupancy Rate', range: [0, 1] },
        height: 300,
        margin: { t: 20, b: 50, l: 50, r: 20 }
    };
    
    Plotly.newPlot('realtimeOccupancyChart', [trace], layout);
}

function loadTopZones() {
    const zones = ['Downtown', 'Midtown', 'Waterfront', 'Shopping', 'Airport'];
    const demand = [0.85, 0.78, 0.72, 0.68, 0.65];
    
    const trace = {
        x: demand,
        y: zones,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: demand,
            colorscale: 'Viridis',
            showscale: false
        }
    };
    
    const layout = {
        xaxis: { title: 'Demand Score', range: [0, 1] },
        height: 300,
        margin: { t: 20, b: 50, l: 100, r: 20 }
    };
    
    Plotly.newPlot('topZonesChart', [trace], layout);
}

function loadPerformanceMetrics() {
    const timePoints = Array.from({length: 60}, (_, i) => i);
    const latency = timePoints.map(() => Math.random() * 100 + 100);
    const accuracy = timePoints.map(() => Math.random() * 0.1 + 0.85);
    
    const trace1 = {
        x: timePoints,
        y: latency,
        type: 'scatter',
        mode: 'lines',
        name: 'Latency (ms)',
        line: { color: '#667eea', width: 2 }
    };
    
    const trace2 = {
        x: timePoints,
        y: accuracy,
        type: 'scatter',
        mode: 'lines',
        name: 'Accuracy',
        yaxis: 'y2',
        line: { color: '#48bb78', width: 2 }
    };
    
    const layout = {
        xaxis: { title: 'Time (minutes)' },
        yaxis: { title: 'Latency (ms)' },
        yaxis2: {
            title: 'Accuracy',
            overlaying: 'y',
            side: 'right',
            range: [0, 1]
        },
        height: 350,
        margin: { t: 20, b: 50, l: 50, r: 50 }
    };
    
    Plotly.newPlot('performanceChart', [trace1, trace2], layout);
}

function loadRecentActivity() {
    const activities = [
        { time: '2 min ago', event: 'User searched parking in Downtown', status: 'success' },
        { time: '5 min ago', event: 'Spot SPOT_12345 became available', status: 'info' },
        { time: '8 min ago', event: 'Payment processed for reservation', status: 'success' },
        { time: '12 min ago', event: 'Traffic update received for Highway 101', status: 'info' },
        { time: '15 min ago', event: 'Model prediction completed with 91% confidence', status: 'success' }
    ];
    
    let html = '<div style="max-height: 300px; overflow-y: auto;">';
    
    activities.forEach(activity => {
        const badgeClass = activity.status === 'success' ? 'badge-success' : 'badge-info';
        html += `
            <div style="padding: 12px; border-bottom: 1px solid #e0e0e0; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #999; font-size: 0.9em;">${activity.time}</span>
                    <p style="margin: 5px 0 0 0; color: #333;">${activity.event}</p>
                </div>
                <span class="badge ${badgeClass}">${activity.status.toUpperCase()}</span>
            </div>
        `;
    });
    
    html += '</div>';
    
    document.getElementById('activityLog').innerHTML = html;
}

function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
}

function startAutoRefresh() {
    updateInterval = setInterval(() => {
        loadDashboardData();
    }, 30000);
}

function stopAutoRefresh() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
});