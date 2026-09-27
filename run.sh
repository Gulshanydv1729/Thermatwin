#!/usr/bin/env bash
# ThermaTwin SRP Digital Twin — Run Script
# Usage:
#   ./run.sh              — Launch Streamlit dual-mode dashboard (default)
#   ./run.sh ingestion    — Launch MQTT → InfluxDB ingestion worker
#   ./run.sh backend      — Launch FastAPI backend
#   ./run.sh frontend     — Launch React/Next.js frontend
#   ./run.sh docker       — Launch full stack via Docker Compose
#   ./run.sh stop         — Stop all Docker containers
#   ./run.sh logs         — Tail Docker container logs
#   ./run.sh migrate      — Run database migrations
#   ./run.sh seed         — Seed database with sample well data
#   ./run.sh test         — Run test suite
#   ./run.sh clean        — Remove Docker volumes and rebuild

set -e

cd "$(dirname "$0")"

MODE="${1:-streamlit}"

case "$MODE" in
    streamlit)
        source venv/bin/activate
        streamlit run app.py
        ;;
    ingestion)
        source venv/bin/activate
        python -m core.ingestion_worker
        ;;
    backend)
        source venv/bin/activate
        uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    frontend)
        cd frontend
        npm install
        npm run dev
        ;;
    docker)
        docker-compose up --build
        ;;
    stop)
        docker-compose down
        ;;
    logs)
        docker-compose logs -f
        ;;
    migrate)
        source venv/bin/activate
        alembic upgrade head
        ;;
    seed)
        source venv/bin/activate
        python -m backend.db.seed
        ;;
    test)
        source venv/bin/activate
        pytest tests/ -v
        ;;
    clean)
        docker-compose down -v
        docker-compose up --build
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run.sh [streamlit|ingestion|backend|frontend|docker|stop|logs|migrate|seed|test|clean]"
        exit 1
        ;;
esac
