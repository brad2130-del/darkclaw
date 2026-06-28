#!/bin/bash
# Darkclaw one-line installer
# Usage: bash install.sh
# Or:    curl -fsSL https://raw.githubusercontent.com/brad2130-del/darkclaw/main/install.sh | bash
set -e

REPO="https://github.com/brad2130-del/darkclaw"
INSTALL_DIR="${DARKCLAW_HOME:-$HOME/.local/share/darkclaw}"
BIN_DIR="${HOME}/.local/bin"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/darkclaw"

BOLD='\033[1m'; GREEN='\033[0;32m'; ROSE='\033[0;35m'; RESET='\033[0m'

echo -e "${ROSE}${BOLD}"
echo "  ██████╗  █████╗ ██████╗ ██╗  ██╗ ██████╗██╗      █████╗ ██╗    ██╗"
echo "  ██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝██╔════╝██║     ██╔══██╗██║    ██║"
echo "  ██║  ██║███████║██████╔╝█████╔╝ ██║     ██║     ███████║██║ █╗ ██║"
echo "  ██║  ██║██╔══██║██╔══██╗██╔═██╗ ██║     ██║     ██╔══██║██║███╗██║"
echo "  ██████╔╝██║  ██║██║  ██║██║  ██╗╚██████╗███████╗██║  ██║╚███╔███╔╝"
echo "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ "
echo -e "${RESET}"
echo "  Self-healing AI harness for your business"
echo ""

# ── Check Python ─────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is required. Install it from https://python.org"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(sys.version_info[:2] >= (3,10))")
if [ "$PY_VER" = "False" ]; then
    echo "ERROR: Python 3.10+ required (you have $(python3 --version))"
    exit 1
fi

echo -e "${BOLD}Installing Darkclaw to $INSTALL_DIR${RESET}"
echo ""

# ── Clone or update ────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "→ Updating existing install…"
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "→ Cloning from GitHub…"
    git clone --depth 1 "$REPO" "$INSTALL_DIR"
fi

# ── Create venv + install deps ─────────────────────────────────────────
echo "→ Setting up Python environment…"
python3 -m venv "$INSTALL_DIR/.venv" --upgrade-deps --quiet
"$INSTALL_DIR/.venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt" httpx

# ── Config dir ─────────────────────────────────────────────────────────
mkdir -p "$CONFIG_DIR"

# ── Launcher script ─────────────────────────────────────────────────────
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/darkclaw" <<LAUNCHER
#!/bin/bash
# Darkclaw launcher — installed by install.sh
DARKCLAW_HOME="$INSTALL_DIR"
DARKCLAW_CONFIG="$CONFIG_DIR/darkclaw.env"
DARKCLAW_DB="$CONFIG_DIR/darkclaw.db"
PORT="\${DARKCLAW_PORT:-7430}"
cd "\$DARKCLAW_HOME"
export DARKCLAW_CONFIG DARKCLAW_DB
"\$DARKCLAW_HOME/.venv/bin/python" -m uvicorn ui.server:app \\
    --host 127.0.0.1 --port "\$PORT" --log-level warning &
PID=\$!
sleep 1.5
xdg-open "http://127.0.0.1:\$PORT" 2>/dev/null || \\
  sensible-browser "http://127.0.0.1:\$PORT" 2>/dev/null || \\
  echo "Open your browser at http://127.0.0.1:\$PORT"
wait \$PID
LAUNCHER
chmod +x "$BIN_DIR/darkclaw"

# ── Desktop shortcut ───────────────────────────────────────────────────
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_DIR/darkclaw.desktop" <<DESKTOP
[Desktop Entry]
Name=Darkclaw
Comment=Self-healing AI harness for your business
Exec=$BIN_DIR/darkclaw
Icon=$INSTALL_DIR/build/darkclaw.AppDir/darkclaw.png
Type=Application
Categories=Office;Utility;
DESKTOP

update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

# ── PATH hint ──────────────────────────────────────────────────────────
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo "  Add this to your ~/.bashrc or ~/.zshrc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Darkclaw installed!${RESET}"
echo ""
echo "  Start:   darkclaw"
echo "  Config:  $CONFIG_DIR/"
echo "  Logs:    $CONFIG_DIR/darkclaw.log"
echo ""
echo "  Rosie is waiting. 🌿"
