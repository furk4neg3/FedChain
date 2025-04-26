# /app/blockchain/scripts/deploy.py

from web3 import Web3
from solcx import compile_standard, install_solc
import json, pathlib

install_solc("0.8.0")

w3 = Web3(Web3.HTTPProvider("http://ganache:8545"))
assert w3.is_connected(), "Cannot reach Ganache"

# Use the first Ganache account both as deployer and server identity
w3.eth.default_account = w3.eth.accounts[0]
account = w3.eth.default_account

# Read & compile
ROOT = pathlib.Path("/app/blockchain")
sol = (ROOT / "contracts" / "FederatedLearning.sol").read_text()
compiled = compile_standard({
    "language": "Solidity",
    "sources": {"FederatedLearning.sol": {"content": sol}},
    "settings": {"outputSelection": {"*": {"*": ["abi","evm.bytecode"]}}},
})
abi      = compiled["contracts"]["FederatedLearning.sol"]["FederatedLearning"]["abi"]
bytecode = compiled["contracts"]["FederatedLearning.sol"]["FederatedLearning"]["evm"]["bytecode"]["object"]

# 1️⃣ Deploy
factory = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = factory.constructor().transact()
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
deployed_address = receipt.contractAddress
print("✓ Contract deployed at", deployed_address)

# 2️⃣ Re-bind to the live contract
contract = w3.eth.contract(address=deployed_address, abi=abi)

# 3️⃣ Register this node
tx_reg = contract.functions.registerNode(account).transact({"from": account})
w3.eth.wait_for_transaction_receipt(tx_reg)
print("✓ Server node registered:", account)

# 4️⃣ Write out the info for server to pick up
info = {"abi": abi, "address": deployed_address}
(ROOT / "scripts" / "contract_info.json").write_text(json.dumps(info))
