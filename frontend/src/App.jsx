import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import './index.css';

// Register Chart.js components for use
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

// Use the environment variable for the API URL for production readiness
const API_BASE_URL ='http://localhost:8000';

const App = () => {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [llmResponse, setLlmResponse] = useState(null); // <-- NEW STATE

  // useEffect hook to fetch data when the component first mounts
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/metrics`);
        setMetrics(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch data. Please ensure the FastAPI backend is running.");
        setLoading(false);
        console.error(err);
      }
    };
    fetchData();
  }, []); // Empty dependency array means this effect runs once on mount

  // NEW: Function to handle the LLM API call
  const handleGenerateFix = async (batchId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/generate_fix/${batchId}`);
      setLlmResponse(response.data);
    } catch (err) {
      console.error(err);
      setLlmResponse({error: "Failed to generate fix. Please check the backend logs."});
    }
  };

  // Prepare data for the charts
  const dataDriftChartData = {
    labels: [...new Set(metrics.map(m => m.batch_id))].sort((a, b) => a - b),
    datasets: [
      {
        label: 'MonthlyCharges Drift Score',
        data: metrics.filter(m => m.feature_name === 'MonthlyCharges' && m.metric_type === 'data_drift_psi_num').sort((a, b) => a.batch_id - b.batch_id).map(m => m.drift_score),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'Contract Drift Score',
        data: metrics.filter(m => m.feature_name === 'Contract' && m.metric_type === 'data_drift_psi_cat').sort((a, b) => a.batch_id - b.batch_id).map(m => m.drift_score),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  };

  // Chart options
  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Data Drift Over Time' },
    },
    scales: {
      y: { min: 0 }
    }
  };

  // Filter for alerts to display in the table
  const alerts = metrics.filter(m => m.is_drifted).sort((a, b) => a.timestamp > b.timestamp ? -1 : 1);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-6">DriftRadar Dashboard</h1>
        {loading && <p className="text-lg text-gray-700">Loading metrics...</p>}
        {error && <div className="p-4 bg-red-100 text-red-700 rounded-md">{error}</div>}
        {!loading && !error && (
          <div>
            <div className="bg-white p-6 rounded-lg shadow-md mb-8">
              <h2 className="text-2xl font-semibold mb-4">Data Drift Metrics</h2>
              <Line data={dataDriftChartData} options={options} />
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-md mb-8">
              <h2 className="text-2xl font-semibold mb-4">Recent Alerts</h2>
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Batch ID</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Feature</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Alert</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th> {/* <-- NEW */}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {alerts.map((alert) => (
                    <tr key={alert.id}>
                      <td className="px-6 py-4 whitespace-nowrap">{alert.batch_id}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{new Date(alert.timestamp).toLocaleString()}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{alert.feature_name}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-medium">{alert.alert_message}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 font-medium">
                        <button onClick={() => handleGenerateFix(alert.batch_id)} className="font-semibold text-white bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded-full">
                          Generate Fix
                        </button>
                      </td> {/* <-- NEW BUTTON */}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* NEW: Section to display the LLM's response */}
            {llmResponse && (
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h2 className="text-2xl font-semibold mb-4">LLM Remediation Copilot</h2>
                <div className="font-mono text-gray-800">
                  <h3 className="text-lg font-bold mb-2">Root Cause Analysis</h3>
                  <p className="mb-4">{llmResponse.llm_response.root_cause}</p>
                  
                  <h3 className="text-lg font-bold mb-2">Suggested Fix</h3>
                  <pre className="bg-gray-900 text-green-400 p-4 rounded-md overflow-x-auto">
                    <code>{llmResponse.llm_response.remediation_code}</code>
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
