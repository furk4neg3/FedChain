# Federated_Learning_on_Blockchain
Steps to run the project:

Clone the repo:

**git clone https://github.com/furk4neg3/federated‑learning‑on‑blockchain.git**

**cd federated‑learning‑on‑blockchain**

Set **.env** file (only contains SERVER_IP="XXX.XXX.XXX.XXX". find the needed value using **ipconfig**, and paste it inside.)

write **docker-compose up --build** (most of the times, this will take more than 1 hour. For not repeating it, write **docker-compose down --remove-orphans** after running the project, this way it won't build from scratch when you call docker-compose up --build again.)

And the project is running! It takes long while building, and after build finishes, it takes a while to train 3 local models 3 times and combine it in global model with blockchain. Be patient, it will work. Ignore the Warning messages, they are natural.

# View the HTML Report

Simply open the report in your browser:

## macOS

```bash
open logs/report.html


## Linux

```bash
xdg-open logs/report.html

## Windows PowerShell

```bash
start .\logs\report.html
