#!/usr/bin/env sh
set -eu

BIN_NAME="${PI_RLM_BIN_NAME:-pi-rlm}"
BIN_DIR="${PI_RLM_BIN_DIR:-$HOME/.local/bin}"
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CLI_PATH="$REPO_DIR/packages/coding-agent/dist/cli.js"

if ! command -v node >/dev/null 2>&1; then
	echo "error: node is required (>=22.19.0)" >&2
	exit 1
fi

NODE_VERSION=$(node -p "process.versions.node")
node -e "const [major, minor, patch] = process.versions.node.split('.').map(Number); process.exit(major > 22 || (major === 22 && (minor > 19 || (minor === 19 && patch >= 0))) ? 0 : 1)" || {
	echo "error: node >=22.19.0 is required, found $NODE_VERSION" >&2
	exit 1
}

cd "$REPO_DIR"

echo "Installing dependencies..."
npm install --ignore-scripts

echo "Building pi-rlm without fetching remote model metadata..."
(cd packages/tui && npm run build)
(cd packages/ai && ../../node_modules/.bin/tsgo -p tsconfig.build.json)
(cd packages/agent && npm run build)
(cd packages/coding-agent && npm run build)

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/$BIN_NAME" <<EOF
#!/usr/bin/env sh
exec node "$CLI_PATH" "\$@"
EOF
chmod +x "$BIN_DIR/$BIN_NAME"

echo "Installed $BIN_NAME to $BIN_DIR/$BIN_NAME"
case ":$PATH:" in
	*:"$BIN_DIR":*) ;;
	*)
		echo ""
		echo "Add this to your shell profile if needed:"
		echo "  export PATH=\"$BIN_DIR:\$PATH\""
		;;
esac

echo ""
"$BIN_DIR/$BIN_NAME" --version
