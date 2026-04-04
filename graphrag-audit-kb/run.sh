#!/bin/bash
# GraphRAG Audit KB - One-click Startup Script
# 一键启动脚本：依赖安装 + 服务拉起

set -e

echo "=========================================="
echo "GraphRAG Audit Knowledge Base - Startup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}[3/5] Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}[4/5] Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Copy .env.example to .env if not exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}Please update .env with your actual configuration!${NC}"
fi

# Start Docker services
echo -e "${YELLOW}[5/5] Starting Docker services (Neo4j + Chroma)...${NC}"
docker-compose up -d

# Wait for services to be ready
echo ""
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Health check
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Services Status:${NC}"
echo -e "${GREEN}  - Neo4j: http://localhost:7474 (bolt://localhost:7687)${NC}"
echo -e "${GREEN}  - Chroma: http://localhost:8000${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Update .env with your LLM API key"
echo "  2. Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8080"
echo "  3. Access API docs: http://localhost:8080/docs"
echo "  4. Run tests: pytest tests/ -v"
echo ""
echo -e "${GREEN}Setup complete!${NC}"
