from web3 import Web3
import json, hashlib, pathlib

w3 = Web3(Web3.HTTPProvider("http://ganache:8545"))
assert w3.is_connected(), "Cannot reach Ganache"

info = json.loads(pathlib.Path("/app/blockchain/scripts/contract_info.json").read_text())
contract = w3.eth.contract(address=info["address"], abi=info["abi"])
account  = w3.eth.accounts[0]

def submit_model_hash(model_bytes: bytes):
    h = hashlib.sha256(model_bytes).digest()
    tx = contract.functions.submitModelHash(h).transact({"from": account})
    w3.eth.wait_for_transaction_receipt(tx)
    print("✓ hash submitted", h.hex()[:10], "…")
