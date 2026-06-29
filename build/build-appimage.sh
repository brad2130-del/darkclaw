#!/bin/bash
# Build Darkclaw AppImage
# Requires: appimagetool, python3.11+, pip
# Output:   Darkclaw-x86_64.AppImage
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
APPDIR="$SCRIPT_DIR/darkclaw.AppDir"
VERSION="${DARKCLAW_VERSION:-$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo '1.0.0')}"

echo "=== Darkclaw AppImage Build v$VERSION ==="

# ── Check deps ─────────────────────────────────────────────────────────
for cmd in python3 pip3 appimagetool; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first."
        [ "$cmd" = "appimagetool" ] && echo "  → wget https://github.com/AppImage/appimagetool/releases/latest/download/appimagetool-x86_64.AppImage -O appimagetool && chmod +x appimagetool"
        exit 1
    fi
done

# ── Set up Python env inside AppDir ────────────────────────────────────
PY_PREFIX="$APPDIR/usr"
mkdir -p "$PY_PREFIX/bin" "$PY_PREFIX/lib"

echo "→ Installing dependencies into AppDir…"
python3 -m venv "$APPDIR/.venv" --copies
"$APPDIR/.venv/bin/pip" install --quiet --upgrade pip
"$APPDIR/.venv/bin/pip" install --quiet -r "$ROOT/requirements.txt" httpx

# Wire the venv into usr/
cp -f "$APPDIR/.venv/bin/python3" "$PY_PREFIX/bin/python3" 2>/dev/null || true
rsync -a --quiet "$APPDIR/.venv/lib/" "$PY_PREFIX/lib/" 2>/dev/null || \
  cp -r "$APPDIR/.venv/lib/." "$PY_PREFIX/lib/"

# ── Copy app source ─────────────────────────────────────────────────────
echo "→ Copying source…"
rsync -a --quiet \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/' \
    --exclude='build/' \
    --exclude='darkclaw_project/' \
    "$ROOT/" "$APPDIR/"

# ── Bundle Electron runtime ─────────────────────────────────────────────
ELECTRON_VERSION="33.4.0"
ELECTRON_CACHE="$SCRIPT_DIR/.cache"
ELECTRON_ZIP="$ELECTRON_CACHE/electron-${ELECTRON_VERSION}.zip"
ELECTRON_DIST="$APPDIR/electron"

mkdir -p "$ELECTRON_CACHE"

if [ ! -f "$ELECTRON_ZIP" ]; then
    echo "→ Downloading Electron v${ELECTRON_VERSION} (~80MB, cached after first build)…"
    wget -q --show-progress \
        "https://github.com/electron/electron/releases/download/v${ELECTRON_VERSION}/electron-v${ELECTRON_VERSION}-linux-x64.zip" \
        -O "$ELECTRON_ZIP"
fi

echo "→ Extracting Electron into AppDir…"
# electron/ dir already has main.js + preload.js from the rsync above;
# unzip adds the binary and supporting libs alongside them
unzip -q -o "$ELECTRON_ZIP" -d "$ELECTRON_DIST"
chmod +x "$ELECTRON_DIST/electron"
echo "  Electron ready at $ELECTRON_DIST/electron"

# ── App icon (SVG → PNG via rsvg-convert or Inkscape, fallback to placeholder) ──
ICON_DST="$APPDIR/darkclaw.png"
if [ ! -f "$ICON_DST" ]; then
    echo "→ Generating icon…"
    python3 - <<'PYEOF'
try:
    from PIL import Image, ImageDraw
    img = Image.new('RGBA', (256, 256), (13, 10, 8, 255))
    d = ImageDraw.Draw(img)
    # Draw claw marks
    for ox, ew in [(-40, 2), (0, 3), (40, 2)]:
        d.line([(128+ox, 40), (128+ox//2, 200)], fill=(232,152,144,255), width=ew)
    d.ellipse([116, 28, 140, 52], fill=(232,152,144,255))
    img.save('build/darkclaw.AppDir/darkclaw.png')
    print('  icon generated with Pillow')
except ImportError:
    # Minimal 1x1 placeholder
    import struct, zlib
    def make_png(w, h, r, g, b):
        raw = bytes([0] + [r,g,b,255]*w) * h
        def chunk(tag, data):
            c = zlib.crc32(tag + data) & 0xffffffff
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', c)
        ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    open('build/darkclaw.AppDir/darkclaw.png','wb').write(make_png(64,64,232,152,144))
    print('  icon generated (minimal fallback)')
PYEOF
fi

# ── AppRun must be executable ───────────────────────────────────────────
chmod +x "$APPDIR/AppRun"

# ── Symlinks appimagetool expects ───────────────────────────────────────
[ -L "$APPDIR/.DirIcon" ]   || ln -sf darkclaw.png "$APPDIR/.DirIcon"
[ -L "$APPDIR/AppRun" ]     || true  # already a file

# ── Build AppImage ──────────────────────────────────────────────────────
OUTPUT="$SCRIPT_DIR/Darkclaw-${VERSION}-x86_64.AppImage"
echo "→ Packaging AppImage → $OUTPUT"
ARCH=x86_64 appimagetool "$APPDIR" "$OUTPUT"

chmod +x "$OUTPUT"
echo ""
echo "✓ Done: $OUTPUT"
echo "  Run it:  ./$( basename "$OUTPUT")"
