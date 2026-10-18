#!/bin/bash
# dich install/run/update script

set -e

PACKAGE_NAME="dich"
VENV_DIR=".venv"
PYTHON="${VENV_DIR}/bin/python3"
PIP="${VENV_DIR}/bin/pip"
VERSION_FILE="dich/VERSION"

# Runtime dependencies (README "Dependencies"). mlx-audio runs Whisper on the
# Metal GPU; without it everything transcribes on the CPU.
RUNTIME_DEPS="mlx-lm httpx pypdf python-docx striprtf"
SPEECH_DEPS="sounddevice faster-whisper av numpy"
GPU_DEPS="mlx-audio"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

current_version() {
    cat "${VERSION_FILE}" 2>/dev/null || echo "unknown"
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "python3 not found. Install Python 3.11+ first:"
        echo "  brew install python python-tk   # or the python.org installer"
        exit 1
    fi

    local major minor
    major=$(python3 -c 'import sys; print(sys.version_info[0])')
    minor=$(python3 -c 'import sys; print(sys.version_info[1])')
    if [ "${major}" -lt 3 ] || { [ "${major}" -eq 3 ] && [ "${minor}" -lt 11 ]; }; then
        print_error "Python 3.11+ required, found ${major}.${minor}"
        exit 1
    fi

    if [ "$(uname -m)" != "arm64" ]; then
        print_warning "Not Apple Silicon (arm64): mlx runs, but not on the Metal GPU."
    fi
}

check_venv() {
    if [ ! -d "${VENV_DIR}" ]; then
        print_error "Virtual environment not found at ${VENV_DIR}"
        echo "Create it first with: ./install.sh install"
        exit 1
    fi

    if [ ! -f "${PYTHON}" ]; then
        print_error "Python not found in virtual environment"
        exit 1
    fi
}

enable_hooks() {
    # The local gate: ruff + the fast test suite on every commit (once per clone).
    git config core.hooksPath .githooks 2>/dev/null || true
}

smoke_check() {
    print_status "Checking the installation..."
    if ! ${PYTHON} -c "import tkinter" 2>/dev/null; then
        print_warning "tkinter is missing — the UI will not start."
        echo "  brew install python-tk   # or use the python.org installer"
    fi
    ${PYTHON} -m py_compile dich/__main__.py dich/backend.py dich/ui.py \
        dich/speech_engine.py
    print_status "Core modules compile."
}

install() {
    check_python

    if [ ! -d "${VENV_DIR}" ]; then
        print_status "Creating virtual environment in ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    fi

    print_status "Installing ${PACKAGE_NAME} $(current_version) dependencies..."
    ${PIP} install --upgrade pip
    ${PIP} install ${RUNTIME_DEPS} ${SPEECH_DEPS}

    if [ "${DICH_SKIP_MLX_AUDIO:-0}" = "1" ]; then
        print_warning "Skipping mlx-audio (DICH_SKIP_MLX_AUDIO=1) — STT stays on the CPU."
    else
        print_status "Installing mlx-audio (GPU speech)...  DICH_SKIP_MLX_AUDIO=1 to skip"
        ${PIP} install ${GPU_DEPS} || print_warning "mlx-audio failed — continuing with the CPU speech path."
    fi

    enable_hooks
    smoke_check
    print_status "Installation complete!"
    print_status "You can now run: ./install.sh run   (or .venv/bin/python3 -m dich)"
    print_status "First translation downloads the translation model (~4 GB)."
    print_status "Grant microphone permission in System Settings before recording."
}

run() {
    check_venv
    exec "${PYTHON}" -m dich "$@"
}

dev() {
    check_venv
    print_status "Installing development dependencies..."
    ${PIP} install -r requirements-dev.txt
    enable_hooks
    print_status "Pre-commit gate enabled (ruff + fast test suite on every commit)."
}

update() {
    check_venv
    print_status "Updating ${PACKAGE_NAME} dependencies..."
    ${PIP} install --upgrade ${RUNTIME_DEPS} ${SPEECH_DEPS}
    if [ "${DICH_SKIP_MLX_AUDIO:-0}" != "1" ]; then
        ${PIP} install --upgrade ${GPU_DEPS} || print_warning "mlx-audio upgrade failed — keeping the installed version."
    fi
    print_status "Update complete!"
}

reinstall() {
    check_python
    print_status "Reinstalling ${PACKAGE_NAME}..."
    if [ -d "${VENV_DIR}" ]; then
        rm -rf "${VENV_DIR}"
    fi
    install
}

uninstall() {
    if [ ! -d "${VENV_DIR}" ]; then
        print_warning "Nothing to uninstall: ${VENV_DIR} does not exist."
        return 0
    fi
    if [ "${FORCE:-0}" != "1" ]; then
        read -r -p "Remove ${VENV_DIR} (downloaded models in ~/.config/dich are kept)? [y/N] " reply
        case "${reply}" in
            y|Y) ;;
            *) print_status "Uninstall cancelled."; return 0 ;;
        esac
    fi
    print_status "Removing ${VENV_DIR}..."
    rm -rf "${VENV_DIR}"
    print_status "Uninstallation complete! Models and settings remain in ~/.config/dich."
}

clean() {
    print_status "Cleaning build artifacts and cache files..."
    rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_status "Clean complete! (App data in ~/.config/dich is untouched.)"
}

status() {
    print_status "${PACKAGE_NAME} version: $(current_version)"
    if [ -d "${VENV_DIR}" ]; then
        print_status "Virtual environment: ${VENV_DIR} ($(${PYTHON} --version))"
    else
        print_warning "No virtual environment. Run ./install.sh install"
        return 0
    fi

    print_status "Installed packages:"
    local pkg
    for pkg in mlx-lm mlx-audio faster-whisper sounddevice pypdf python-docx striprtf pytest ruff; do
        if ${PIP} show "${pkg}" >/dev/null 2>&1; then
            echo "  ok       ${pkg} ($(${PIP} show ${pkg} | awk '/^Version:/{print $2}'))"
        else
            echo "  missing  ${pkg}"
        fi
    done

    if ${PYTHON} -c "import tkinter" 2>/dev/null; then
        print_status "tkinter: available"
    else
        print_warning "tkinter: MISSING (brew install python-tk)"
    fi

    if [ "$(git config core.hooksPath 2>/dev/null || true)" = ".githooks" ]; then
        print_status "Pre-commit gate: enabled"
    else
        print_warning "Pre-commit gate: not enabled (./install.sh dev)"
    fi

    if [ -d "${HOME}/.config/dich/models" ]; then
        local size
        size=$(du -sh "${HOME}/.config/dich/models" 2>/dev/null | awk '{print $1}')
        print_status "Model cache: ${HOME}/.config/dich/models (${size})"
    else
        print_status "Model cache: empty (first translation downloads ~4 GB)"
    fi
}

# Main command handler
case "${1:-install}" in
    install)
        install
        ;;
    run)
        run "${@:2}"
        ;;
    dev)
        dev
        ;;
    uninstall)
        uninstall
        ;;
    update)
        update
        ;;
    reinstall)
        reinstall
        ;;
    clean)
        clean
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {install|run|dev|uninstall|update|reinstall|clean|status}"
        echo ""
        echo "Commands:"
        echo "  install    - Create the venv and install all dependencies (default)"
        echo "  run        - Launch the app (extra args go to python -m dich)"
        echo "  dev        - Install dev dependencies and enable the pre-commit gate"
        echo "  uninstall  - Remove the venv (models in ~/.config/dich are kept)"
        echo "  update     - Upgrade all dependencies"
        echo "  reinstall  - Delete the venv and install fresh"
        echo "  clean      - Remove build artifacts and cache files"
        echo "  status     - Show version, packages, tkinter, hooks and model cache"
        echo ""
        echo "Environment:"
        echo "  DICH_SKIP_MLX_AUDIO=1   install/upgrade without the GPU speech stack"
        echo "  FORCE=1                 uninstall without the confirmation prompt"
        exit 1
        ;;
esac
