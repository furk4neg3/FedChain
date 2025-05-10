#!/usr/bin/env python3
import os
import pickle
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from blockchain_interface import submit_model_hash
from model import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import json
import atexit
import webbrowser

# ▶ LOG DIRECTORY SETUP
LOG_DIR = "logs"
# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# ▶ GLOBAL STORAGE FOR AI METRICS
ai_results = []

# Number of federated clients
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

# Prepare MNIST test set
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def evaluate_global(model: Net) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 100.0 * correct / total


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        parameters, metrics = super().aggregate_fit(rnd, results, failures)
        
        # Collect client accuracies
        client_metrics = []
        for client_proxy, fit_res in results:
            acc = None
            if fit_res.metrics and "accuracy" in fit_res.metrics:
                acc = fit_res.metrics["accuracy"] * 100
            cid = getattr(client_proxy, "cid", None) or getattr(client_proxy, "id", None) or str(client_proxy)
            client_metrics.append({"client": cid, "accuracy": acc})
        
        # Submit model hash to blockchain
        ndarrays = parameters_to_ndarrays(parameters)
        serialized = pickle.dumps([arr.tolist() for arr in ndarrays])
        submit_model_hash(serialized)
        
        # Evaluate global model
        global_model = Net()
        state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), ndarrays)}
        global_model.load_state_dict(state_dict, strict=True)
        global_acc = evaluate_global(global_model)
        print(f"[Server] Global Accuracy after round {rnd}: {global_acc:.2f}%")
        
        # Record metrics
        ai_results.append({
            "round": rnd,
            "global_accuracy": global_acc,
            "clients": client_metrics,
        })
        
        # Save the updated global model
        torch.save(global_model.state_dict(), "logs/final_global_model.pth")
        return parameters, metrics


def _write_and_launch_report():
    # 1) Write JSON files
    ai_path = os.path.join(LOG_DIR, "ai_results.json")
    with open(ai_path, "w") as f:
        json.dump(ai_results, f, indent=2)

    bc_path = os.path.join(LOG_DIR, "blockchain_log.json")
    try:
        with open(bc_path) as f:
            bc_logs = json.load(f)
    except Exception:
        bc_logs = []

    # 2) Inline JSON into dark-themed HTML with side-by-side layout
    ai_json = json.dumps(ai_results)
    bc_json = json.dumps(bc_logs)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Federated Learning & Blockchain Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ background-color: #121212; color: #e0e0e0; font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
  header {{ text-align: center; margin-bottom: 30px; }}
  .container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .panel {{ flex: 1; background-color: #1e1e1e; padding: 20px; border-radius: 8px; min-width: 300px; }}
  h1 {{ margin: 0; font-size: 24px; }}
  p {{ margin-top: 8px; color: #b0b0b0; font-size: 14px; }}
  h2 {{ margin-top: 0; font-size: 18px; }}
  canvas {{ width: 100% !important; height: 300px !important; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ border: 1px solid #333; padding: 8px; font-size: 13px; }}
  th {{ background-color: #2a2a2a; }}
  td {{ background-color: #1e1e1e; color: #e0e0e0; }}
</style>
</head>
<body>
<header>
  <h1>Federated Learning on Blockchain</h1>
  <p>A concise professional dashboard showing model accuracy trends and on-chain commitments per round.</p>
</header>
<div class="container">
  <div class="panel">
    <h2>Accuracy Trends</h2>
    <canvas id="flChart"></canvas>
  </div>
  <div class="panel">
    <h2>Blockchain Transactions</h2>
    <table id="bcTable">
      <thead><tr><th>Block</th><th>Tx Hash</th><th>Model Hash</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<script>
const ai = {ai_json};
const bc = {bc_json};
const labels = ai.map(r => `Round ${{r.round}}`);
const datasets = [{{ label: 'Global', data: ai.map(r => r.global_accuracy) }}];
const clients = [...new Set(ai.flatMap(r => r.clients.map(c => c.client)))];
clients.forEach(cid => {{
  datasets.push({{
    label: cid,
    data: ai.map(r => {{ const entry = r.clients.find(c => c.client === cid); return entry ? entry.accuracy : null; }})
  }});
}});
new Chart(document.getElementById('flChart'), {{
  type: 'line',
  data: {{ labels, datasets }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Accuracy Trend', color: '#e0e0e0' }}, legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
    scales: {{ x: {{ ticks: {{ color: '#e0e0e0' }} }}, y: {{ ticks: {{ color: '#e0e0e0' }} }} }}
  }}
}});
const tbody = document.querySelector('#bcTable tbody');
bc.forEach(entry => {{
  const tr = document.createElement('tr');
  ['blockNumber','transactionHash','modelHash'].forEach(key => {{
    const td = document.createElement('td');
    td.textContent = entry[key];
    tr.appendChild(td);
  }});
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"""

    # 3) Write HTML report
    report_path = os.path.join(LOG_DIR, "report.html")
    with open(report_path, "w") as f:
        f.write(html)

    # Attempt to open in default browser (optional)
    try:
        webbrowser.open("file://" + os.path.abspath(report_path))
    except:
        pass

# Register exit hook
atexit.register(_write_and_launch_report)

if __name__ == "__main__":
    strategy = CustomFedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
