// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    address public owner;
    mapping(address => bool) public nodes;
    bytes32[] public modelUpdates;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    function registerNode(address node) external onlyOwner {
        nodes[node] = true;
    }

    function submitModelHash(bytes32 modelHash) external {
        require(nodes[msg.sender], "Node not registered");
        modelUpdates.push(modelHash);
    }

    function getModelHashes() external view returns(bytes32[] memory) {
        return modelUpdates;
    }
}
