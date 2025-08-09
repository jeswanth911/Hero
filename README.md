# AI Data Analytics Platform

## Project Overview

This AI Data Analytics Platform is a scalable, enterprise-grade solution designed to ingest, clean, analyze, and visualize large volumes of heterogeneous data. Leveraging advanced AI models and modern data engineering best practices, it enables organizations to unlock actionable insights through natural language queries, automated data workflows, and customizable dashboards.

### Vision

To empower businesses with seamless, intelligent data analytics capabilities—transforming raw data into strategic knowledge rapidly and reliably, while supporting extensibility and integration across diverse data ecosystems.

---

## Features & Architecture Summary

* **Multi-format Data Ingestion:** Supports CSV, Excel, JSON, XML, PDF tables, and database dumps.
* **Automated Data Cleaning & Normalization:** Enterprise-grade cleaning pipelines to ensure data quality.
* **Natural Language Querying:** AI-powered interface for SQL generation and data Q\&A.
* **Dynamic SQLite Database Conversion:** Converts ingested data into queryable SQLite databases.
* **Visualization & Reporting:** Interactive tables, charts, and exportable reports.
* **Extensible Modular Design:** Clear separation of ingestion, processing, analysis, and serving layers.
* **Asynchronous & Scalable:** Supports batch processing and concurrent workflows.
* **Production-ready Deployment:** Containerized with Docker, orchestrated with docker-compose.

---

## Setup Instructions

### Prerequisites

* Python 3.10+
* Docker & docker-compose
* Git

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
DATABASE_URL=sqlite:///./data/analytics.db
LOG_LEVEL=INFO
PORT=8000
```
