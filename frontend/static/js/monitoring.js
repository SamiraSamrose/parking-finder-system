const API_BASE_URL = 'http://localhost:5000/api';
let monitoringInterval = null;
let isMonitoring = false;

function startMonitoring() {
    if (isMonitoring) {
        showAlert('Monitoring already running', 'info');
        return;
    }
    
    isMonitoring = true;
    updateMetricsSummary();
    
    monitoringInterval = setInterval(() => {
        updateMetricsSummary();
    }, 5000);
    
    showAlert('Monitoring started', 'success');
}

function stopMonitoring() {
    if (!isMonitoring) {
        showAlert('Monitoring not running', 'info');
        return;
    }
    
    isMonitoring = false;
    
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
    }
    
    showAlert('Monitoring stopped', 'success');
}

async function simulateMonitoring() {
    showLoadingIndicator('Simulating monitoring data...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/monitoring/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                duration_minutes: 60,
                samples_per_minute: 120
            })
        });
        
        const data = await response.json();
        
        displayLLMPerformance(data.data);
        displayPredictionAccuracy();
        
        showAlert(`Simulated ${data.total_samples} monitoring data points`, 'success');
    } catch (error) {
        console.error('Simulation error:', error);
        showAlert(`Simulation failed: ${error.message}`, 'error');
    } finally {
        hideLoadingIndicator();
    }
}

function updateMetricsSummary() {
    const uptime = calculateUptime();
    const totalRequests = Math.floor(Math.random() * 10000) + 50000;
    const successRate = (Math.random() * 5 + 95).toFixed(2);
    const avgLatency = (Math.random() * 100 + 150).toFixed(0);
    
    document.getElementById('uptime').textContent = uptime;
    document.getElementById('totalRequests').textContent = totalRequests.toLocaleString();
    document.getElementById('successRate').textContent = `${successRate}%`;
    document.getElementById('avgLatency').textContent = `${avgLatency}ms`;
}

function calculateUptime() {
    const startTime = Date.now() - (Math.random() * 86400000);
    const uptime = Date.now() - startTime;
    
    const hours = Math.floor(uptime / 3600000);
    const minutes = Math.floor((uptime % 3600000) / 60000);
    
    return `${hours}h ${minutes}m`;
}

function displayLLMPerformance(metricsData) {
    const timestamps = metricsData.timestamps.map(t => new Date(t * 1000));
    const latencies = metricsData.latencies;
    const tokenUsage = metricsData.token_usage;
    const successRates = metricsData.success_rates;
    
    const trace1 = {
        x: timestamps,
        y: latencies,
        type: 'scatter',
        mode: 'lines',
        name: 'Latency',
        line: { color: '#667eea', width: 2 }
    };
    
    const layout1 = {
        title: 'LLM Latency Over Time',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Latency (ms)' },
        height: 300
    };
    
    Plotly.newPlot('llmLatencyChart', [trace1], layout1);
    
    const trace2 = {
        x: timestamps,
        y: tokenUsage,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: { color: '#48bb78', width: 2 }
    };
    
    const layout2 = {
        title: 'LLM Token Usage Over Time',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Tokens' },
        height: 300
    };
    
    Plotly.newPlot('llmTokenUsageChart', [trace2], layout2);
    
    const windowSize = 100;
    const successRateSmoothed = [];
    
    for (let i = windowSize; i < successRates.length; i++) {
        const window = successRates.slice(i - windowSize, i);
        const avg = window.reduce((a, b) => a + b, 0) / window.length;
        successRateSmoothed.push(avg * 100);
    }
    
    const trace3 = {
        x: timestamps.slice(windowSize),
        y: successRateSmoothed,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ed8936', width: 2 }
    };
    
    const layout3 = {
        title: 'LLM Success Rate Over Time',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Success Rate (%)', range: [90, 100] },
        height: 300
    };
    
    Plotly.newPlot('llmSuccessRateChart', [trace3], layout3);
}


function displayPredictionAccuracy() {
    const timePoints = Array.from({length: 60}, (_, i) => i);
    const accuracy = timePoints.map(() => Math.random() * 0.1 + 0.85);
    
    const trace1 = {
        x: timePoints,
        y: accuracy,
        type: 'scatter',
        mode:'lines+markers',
        line: { color: '#667eea', width: 3 },
        marker: { size: 6 }
    };
    
    const layout1 = {
        title: 'Prediction Accuracy Over Time',
        xaxis: { title: 'Time (minutes)' },
        yaxis: { title: 'Accuracy', range: [0.7, 1.0] },
        height: 350
    };
    
    Plotly.newPlot('accuracyOverTimeChart', [trace1], layout1);
    
    const confidenceValues = Array.from({length: 1000}, () => Math.random() * 0.3 + 0.7);
    
    const trace2 = {
        x: confidenceValues,
        type: 'histogram',
        nbinsx: 30,
        marker: { color: '#48bb78' }
    };
    
    const layout2 = {
        title: 'Confidence Score Distribution',
        xaxis: { title: 'Confidence Score' },
        yaxis: { title: 'Frequency' },
        height: 350
    };
    
    Plotly.newPlot('confidenceDistributionChart', [trace2], layout2);
}


async function testHallucinationDetection() {
    showLoadingIndicator('Testing hallucination detection...');
    
    try {
        const testCases = [
            {
                spot_id: 'SPOT_12345',
                restricted_zones: ['downtown_main'],
                restriction_status: 'Temporarily Restricted',
                restriction_reason: 'street_cleaning'
            },
            {
                spot_id: 'SPOT_67890',
                restricted_zones: [],
                restriction_status: 'Available',
                restriction_reason: 'none'
            },
            {
                spot_id: 'SPOT_downtown_main_001',
                restricted_zones: ['downtown_main'],
                restriction_status: 'Temporarily Restricted',
                restriction_reason: 'special_event'
            }
        ];
        
        const results = [];
        
        for (const testCase of testCases) {
            const response = await fetch(`${API_BASE_URL}/monitoring/detect-hallucination`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(testCase)
            });
            
            const result = await response.json();
            results.push({
                ...testCase,
                is_hallucination: result.is_hallucination
            });
        }
        
        displayHallucinationResults(results);
        
        const scores = Array.from({length: 100}, () => Math.random());
        
        const trace = {
            x: Array.from({length: 100}, (_, i) => i),
            y: scores,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#f56565', width: 2 },
            marker: { 
                size: 6,
                color: scores.map(s => s > 0.7 ? '#f56565' : '#48bb78')
            }
        };
        
        const layout = {
            title: 'Hallucination Detection Scores',
            xaxis: { title: 'Request Number' },
            yaxis: { title: 'Hallucination Score', range: [0, 1] },
            height: 350,
            shapes: [{
                type: 'line',
                x0: 0,
                x1: 100,
                y0: 0.7,
                y1: 0.7,
                line: { color: '#f56565', width: 2, dash: 'dash' }
            }]
        };
        
        Plotly.newPlot('hallucinationScoreChart', [trace], layout);
        
        showAlert('Hallucination detection test complete', 'success');
    } catch (error) {
        console.error('Hallucination detection error:', error);
        showAlert(`Test failed: ${error.message}`, 'error');
    } finally {
        hideLoadingIndicator();
    }
}


function displayHallucinationResults(results) {
    const container = document.getElementById('hallucinationEventsTable');
    
    let html = '<table><thead><tr>';
    html += '<th>Spot ID</th><th>Status</th><th>Reason</th><th>Hallucination Detected</th>';
    html += '</tr></thead><tbody>';
    
    results.forEach(result => {
        html += '<tr>';
        html += `<td>${result.spot_id}</td>`;
        html += `<td>${result.restriction_status}</td>`;
        html += `<td>${result.restriction_reason}</td>`;
        html += `<td><span class="badge ${result.is_hallucination ? 'badge-danger' : 'badge-success'}">${result.is_hallucination ? 'YES' : 'NO'}</span></td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    container.innerHTML = html;
}


async function testPromptInjection() {
    const userInput = document.getElementById('testPromptInput').value;
    
    if (!userInput) {
        showAlert('Please enter text to test', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/monitoring/detect-injection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_input: userInput
            })
        });
        
        const result = await response.json();
        
        displayInjectionResults(result, userInput);
        
        if (result.is_injection) {
            showAlert('Prompt injection detected!', 'danger');
        } else {
            showAlert('No prompt injection detected', 'success');
        }
    } catch (error) {
        console.error('Injection detection error:', error);
        showAlert(`Detection failed: ${error.message}`, 'error');
    }
}


function displayInjectionResults(result, userInput) {
    const container = document.getElementById('injectionResults');
    
    let html = '<div class="metrics-container">';
    html += '<h4>Detection Results</h4>';
    html += `<div class="metric-row"><span>Input:</span><span>${userInput.substring(0, 100)}${userInput.length > 100 ? '...' : ''}</span></div>`;
    html += `<div class="metric-row"><span>Injection Detected:</span><span class="badge ${result.is_injection ? 'badge-danger' : 'badge-success'}">${result.is_injection ? 'YES' : 'NO'}</span></div>`;
    html += `<div class="metric-row"><span>Confidence:</span><span>${(result.confidence * 100).toFixed(1)}%</span></div>`;
    
    if (result.detected_patterns && result.detected_patterns.length > 0) {
        html += `<div class="metric-row"><span>Detected Patterns:</span><span>${result.detected_patterns.join(', ')}</span></div>`;
    }
    
    html += '</div>';
    
    container.innerHTML = html;
    
    const timestamps = Array.from({length: 50}, (_, i) => i);
    const injectionAttempts = timestamps.map(() => Math.random() > 0.9 ? 1 : 0);
    
    const trace = {
        x: timestamps,
        y: injectionAttempts,
        type: 'scatter',
        mode: 'markers',
        marker: {
            size: 10,
            color: injectionAttempts.map(v => v === 1 ? '#f56565' : '#48bb78')
        }
    };
    
    const layout = {
        title: 'Security Events Timeline',
        xaxis: { title: 'Request Number' },
        yaxis: { title: 'Injection Attempt', tickvals: [0, 1], ticktext: ['No', 'Yes'] },
        height: 300
    };
    
    Plotly.newPlot('securityEventsChart', [trace], layout);
}


async function simulateStream() {
    const streamType = document.getElementById('streamType').value;
    
    showLoadingIndicator(`Simulating ${streamType} stream...`);
    
    try {
        const response = await fetch(`${API_BASE_URL}/kafka/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                stream_type: streamType,
                duration_seconds: 60
            })
        });
        
        const data = await response.json();
        
        displayStreamMetrics(data);
        
        showAlert(`Simulated ${data.events_produced} ${streamType} events`, 'success');
    } catch (error) {
        console.error('Stream simulation error:', error);
        showAlert(`Simulation failed: ${error.message}`, 'error');
    } finally {
        hideLoadingIndicator();
    }
}


function displayStreamMetrics(data) {
    const timePoints = Array.from({length: 60}, (_, i) => i);
    const throughput = timePoints.map(() => Math.random() * 5000 + 10000);
    
    const trace1 = {
        x: timePoints,
        y: throughput,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: { color: '#667eea', width: 2 }
    };
    
    const layout1 = {
        title: 'Stream Throughput (messages/sec)',
        xaxis: { title: 'Time (seconds)' },
        yaxis: { title: 'Messages/Second' },
        height: 300
    };
    
    Plotly.newPlot('streamThroughputChart', [trace1], layout1);
    
    const latency = timePoints.map(() => Math.random() * 50 + 20);
    
    const trace2 = {
        x: timePoints,
        y: latency,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#48bb78', width: 2 },
        marker: { size: 5 }
    };
    
    const layout2 = {
        title: 'Stream Processing Latency (ms)',
        xaxis: { title: 'Time (seconds)' },
        yaxis: { title: 'Latency (ms)' },
        height: 300
    };
    
    Plotly.newPlot('streamLatencyChart', [trace2], layout2);
}


async function simulateVoiceQueries() {
    showLoadingIndicator('Simulating voice queries...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/voice/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_queries: 10
            })
        });
        
        const data = await response.json();
        
        displayVoiceMetrics(data);
        
        showAlert(`Simulated ${data.total_queries} voice queries`, 'success');
    } catch (error) {
        console.error('Voice simulation error:', error);
        showAlert(`Simulation failed: ${error.message}`, 'error');
    } finally {
        hideLoadingIndicator();
    }
}


function displayVoiceMetrics(data) {
    const agents = Object.keys(data.agent_distribution);
    const counts = Object.values(data.agent_distribution);
    
    const trace1 = {
        labels: agents,
        values: counts,
        type: 'pie',
        marker: {
            colors: ['#667eea', '#48bb78', '#ed8936', '#4299e1']
        }
    };
    
    const layout1 = {
        title: 'Agent Distribution',
        height: 300
    };
    
    Plotly.newPlot('agentDistributionChart', [trace1], layout1);
    
    const latencies = data.conversations.map(c => c.processing_time_ms);
    
    const trace2 = {
        y: latencies,
        type: 'box',
        marker: { color: '#667eea' }
    };
    
    const layout2 = {
        title: 'Voice Query Latency Distribution',
        yaxis: { title: 'Latency (ms)' },
        height: 300
    };
    
    Plotly.newPlot('voiceLatencyChart', [trace2], layout2);
}


function showLoadingIndicator(message = 'Loading...') {
    const indicator = document.createElement('div');
    indicator.id = 'loadingIndicator';
    indicator.innerHTML = `<div class="loading"></div><p>${message}</p>`;
    indicator.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:white;padding:30px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,0.1);text-align:center;z-index:1000;';
    document.body.appendChild(indicator);
}


function hideLoadingIndicator() {
    const indicator = document.getElementById('loadingIndicator');
    if (indicator) {
        indicator.remove();
    }
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
    console.log('Monitoring Dashboard loaded');
    updateMetricsSummary();
});