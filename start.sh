#!/bin/bash
#cd /Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis
#./start.sh

# ─────────────────────────────────────────────
#  AuthBrain AI Face Analysis — Start Script
# ─────────────────────────────────────────────

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

echo ""
echo "🧠 AuthBrain AI Face Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Kill anything on ports 3000 & 8000 ──────
echo "🔴 Stopping any existing sessions..."
lsof -ti:3000,8000 | xargs kill -9 2>/dev/null
sleep 1
echo "✅ Ports 3000 & 8000 are free"

# ── Mode Selection ───────────────────────────
MODE="web"
if [ "$1" == "--mac" ] || [ "$1" == "-m" ]; then
  MODE="mac"
fi

# ── Start Backend ────────────────────────────
echo ""
echo "🚀 Starting Backend (FastAPI) on http://localhost:8000 ..."
cd "$BACKEND_DIR"
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# ── Start Frontend / macOS Desktop ───────────
echo ""
cd "$FRONTEND_DIR"
if [ "$MODE" == "mac" ]; then
  echo "🍏 Starting macOS Desktop Application (Electron)..."
  npx vite --host 0.0.0.0 --port 3000 --strictPort &
  FRONTEND_PID=$!
  echo "   Vite Dev PID: $FRONTEND_PID"
  npx wait-on http://localhost:3000
  npx electron . &
  ELECTRON_PID=$!
  echo "   Electron PID: $ELECTRON_PID"
else
  echo "🌐 Starting Frontend (Vite/React) on http://localhost:3000 ..."
  npx vite --host 0.0.0.0 --port 3000 --strictPort &
  FRONTEND_PID=$!
  echo "   Frontend PID: $FRONTEND_PID"
fi

# ── Summary ──────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Both services started!"
echo "   Frontend → http://localhost:3000"
echo "   Backend  → http://localhost:8000"
echo "   API Docs → http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Press Ctrl+C to stop all services."
echo ""

# ── Wait & cleanup on Ctrl+C ─────────────────
trap "echo ''; echo '🔴 Stopping all services...'; kill $BACKEND_PID $FRONTEND_PID $ELECTRON_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
