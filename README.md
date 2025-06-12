# EnterpriseMLwithDocker 🚀

A modular framework showcasing how to build, train, and deploy machine learning (ML) models within Dockerized enterprise-ready pipelines.

## 📌 Overview

This repository demonstrates best practices for:
- Containerizing Python-based ML workflows.
- Structuring code for training, evaluation, and inference.
- Integrating CI/CD for automated testing and deployment via Docker.

The goal is to provide a foundation for building scalable, production-grade ML systems in an enterprise environment.

---

## 🧩 Repository Structure
enterprisemlwithdocker/
├── data/ # Sample datasets
├── src/ # Core Python modules
│ ├── models/ # Model definitions
│ ├── data.py # Data loading/preprocessing
│ ├── train.py # Training scripts
│ └── inference.py # Deployment / inference logic
├── Dockerfile # Base Docker container for training
├── docker-compose.yml # Orchestrates training and inference services
├── requirements.txt # Python dependencies
├── tests/ # Unit tests for key modules
└── README.md # This file

---

## ⚙️ Requirements

- Docker & Docker Compose installed
- Python 3.9+
- (Optional) GPU support for heavy model training

---

## 🚀 Quick Start

### Build the containers

```bash
docker-compose build

docker-compose run ml python -m pytest -q

docker-compose run ml python src/train.py --epochs 10 --data-path /data/train.csv

docker-compose up

