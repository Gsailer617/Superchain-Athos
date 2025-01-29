import fs from 'fs';
import path from 'path';

// Read the contract ABI from the artifacts
const artifactPath = path.join(__dirname, '../artifacts/contracts/FlashLoanArbitrage.sol/FlashLoanArbitrage.json');
const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));

// Extract the ABI
const abi = artifact.abi;

// Write the ABI to a new file
const abiPath = path.join(__dirname, '../abi/FlashLoanArbitrage.json');

// Create the abi directory if it doesn't exist
if (!fs.existsSync(path.join(__dirname, '../abi'))) {
    fs.mkdirSync(path.join(__dirname, '../abi'));
}

// Write the ABI file
fs.writeFileSync(abiPath, JSON.stringify(abi, null, 2));

console.log('ABI file generated successfully at:', abiPath); 