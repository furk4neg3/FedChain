from web3 import Web3
from solcx import compile_standard
import json

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
w3.eth.default_account = w3.eth.accounts[0]

with open("../contracts/FederatedLearning.sol", "r") as file:
    contract_source = file.read()

compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {"FederatedLearning.sol": {"content": contract_source}},
    "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}}}
})

bytecode = compiled_sol['contracts']['FederatedLearning.sol']['FederatedLearning']['evm']['bytecode']['object']
abi = compiled_sol['contracts']['FederatedLearning.sol']['FederatedLearning']['abi']

contract = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = contract.constructor().transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"Contract deployed at address: {tx_receipt.contractAddress}")

with open('contract_info.json', 'w') as file:
    json.dump({"abi": abi, "address": tx_receipt.contractAddress}, file)
