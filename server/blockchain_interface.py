from web3 import Web3
import json, hashlib, pathlib, os

# ▶ LOG DIRECTORY SETUP (match server.py)
LOG_DIR = os.getenv("LOG_DIR", "logs")
# If running inside container, ensure logs folder exists
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize Web3 connection
w3 = Web3(Web3.HTTPProvider("http://ganache:8545"))
assert w3.is_connected(), "Cannot reach Ganache"

# Load contract info
info = json.loads(pathlib.Path("/app/blockchain/scripts/contract_info.json").read_text())
contract = w3.eth.contract(address=info["address"], abi=info["abi"])
account = w3.eth.accounts[0]

# Path to blockchain log (inside logs folder)
LOG_PATH = pathlib.Path(LOG_DIR) / "blockchain_log.json"


def submit_model_hash(model_bytes: bytes):
    # Compute SHA-256
    h = hashlib.sha256(model_bytes).digest()
    tx = contract.functions.submitModelHash(h).transact({"from": account})

    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx)

    # Build entry
    entry = {
        "blockNumber": receipt.blockNumber,
        "transactionHash": receipt.transactionHash.hex(),
        "modelHash": h.hex()
    }

    # Append or create blockchain log
    try:
        logs = json.loads(LOG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    logs.append(entry)
    LOG_PATH.write_text(json.dumps(logs, indent=2))

    print(f"✓ hash submitted {h.hex()[:10]}…")
    return receipt
