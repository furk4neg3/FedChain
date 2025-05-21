# FedChain 🚀

A privacy-preserving federated-learning system secured by an Ethereum blockchain.  
Clients train on disjoint MNIST shards, submit hashed model updates via Solidity smart contracts, and store both submissions and the aggregated global model on-chain.  

---

## 🔧 Prerequisites

- Git  
- Docker & Docker Compose  
- A local Ethereum network (we use Ganache CLI)  

---

## ⚙️ Setup & Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/furk4neg3/FedChain.git
   cd FedChain
   ```

2. **Create your `.env`**  
   ```env
   SERVER_IP="XXX.XXX.XXX.XXX"
   ```  
   > Find your IP via:
   > - **Windows:** `ipconfig`  
   > - **macOS/Linux:** `ifconfig` or `hostname -I`

3. **Build & start everything**  
   ```bash
   docker-compose up --build
   ```
   - 🚨 **Heads-up:** The first build can take **over 1 hour**, as it builds images *and* runs 3 rounds of local training + global aggregation on MNIST.  
   - To avoid rebuilding from scratch on your next run:
     ```bash
     docker-compose down --remove-orphans
     docker-compose up --build
     ```

---

## 📊 What’s Happening Under the Hood

1. **Local Training**  
   - 3 clients each train a small CNN on their shard (C1: digits 0–2; C2: 3–6; C3: 7–9).  
   - Local accuracies: ~30 %, ~40 %, ~30 %.  
2. **Submission**  
   - Each client serializes its weights → computes a Keccak hash → calls `submitModel(hash, accuracy)` on the smart contract.  
3. **On-Chain Logging**  
   - Solidity contract (on Ganache) logs `(round, client, hash, accuracy)` in a new block.  
4. **Server Aggregation**  
   - Flask+Web3.py service listens for submission events → fetches off-chain weights → runs Federated Averaging → ~85 % global accuracy → calls `updateGlobalWeights(...)`.  
5. **Dashboard Updates**  
   - Flask + Socket.IO pushes real-time charts (Chart.js) showing local vs. global accuracy and block/transaction metadata.

---

## 👀 View the HTML Report

After the containers are up & training finishes, open:

### macOS
```bash
open logs/report.html
```

### Linux
```bash
xdg-open logs/report.html
```

### Windows PowerShell
```powershell
start .\logs\report.html
```

---

Happy federating! 🤝🔒
