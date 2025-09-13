#!/bin/bash

# Copy required files from Desktop to pipecat-quickstart project
# This script copies the necessary kokoro and moonshine files that are not yet in the official release

set -e  # Exit on any error

echo "🚀 Copying required files from Desktop..."

# Define paths
DESKTOP_FOLDER="$HOME/Desktop/DO_NOT_DELETE_Copy_this_to_the_pipecat-quickstart_project"
PROJECT_PATH="$(pwd)"
VENV_PATH="$PROJECT_PATH/venv/lib/python3.12/site-packages/pipecat/services"

# Check if Desktop folder exists
if [ ! -d "$DESKTOP_FOLDER" ]; then
    echo "❌ Error: Desktop folder not found at:"
    echo "    $DESKTOP_FOLDER"
    exit 1
fi

# Check if required subfolders exist
if [ ! -d "$DESKTOP_FOLDER/assets" ]; then
    echo "❌ Error: $DESKTOP_FOLDER/assets not found"
    exit 1
fi

# The kokoro and moonshine folders are nested in the venv structure
DESKTOP_SERVICES="$DESKTOP_FOLDER/venv/lib/python3.12/site-packages/pipecat/services"
DESKTOP_PACKAGES="$DESKTOP_FOLDER/venv/lib/python3.12/site-packages"

if [ ! -d "$DESKTOP_SERVICES/kokoro" ]; then
    echo "❌ Error: $DESKTOP_SERVICES/kokoro not found"
    exit 1
fi

if [ ! -d "$DESKTOP_SERVICES/moonshine" ]; then
    echo "❌ Error: $DESKTOP_SERVICES/moonshine not found"
    exit 1
fi

# Check for additional required packages
if [ ! -d "$DESKTOP_PACKAGES/kokoro_onnx" ]; then
    echo "❌ Error: $DESKTOP_PACKAGES/kokoro_onnx not found"
    exit 1
fi

if [ ! -d "$DESKTOP_PACKAGES/moonshine_onnx" ]; then
    echo "❌ Error: $DESKTOP_PACKAGES/moonshine_onnx not found"
    exit 1
fi

if [ ! -d "$DESKTOP_PACKAGES/kokoro_onnx-0.4.9.dist-info" ]; then
    echo "❌ Error: $DESKTOP_PACKAGES/kokoro_onnx-0.4.9.dist-info not found"
    exit 1
fi

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "Please make sure you've installed the requirements first: pip install -r requirements.txt"
    exit 1
fi

echo "📁 Copying assets folder..."
# Copy assets folder (merge/overwrite)
if [ -d "$PROJECT_PATH/assets" ]; then
    echo "   Backing up existing assets to assets.backup..."
    cp -r "$PROJECT_PATH/assets" "$PROJECT_PATH/assets.backup"
fi

cp -r "$DESKTOP_FOLDER/assets" "$PROJECT_PATH/"
echo "✅ Assets folder copied successfully"

echo "📁 Updating kokoro service..."
# Remove existing kokoro folder and copy new one
if [ -d "$VENV_PATH/kokoro" ]; then
    echo "   Removing existing kokoro folder..."
    rm -rf "$VENV_PATH/kokoro"
fi

cp -r "$DESKTOP_SERVICES/kokoro" "$VENV_PATH/"
echo "✅ Kokoro service updated successfully"

echo "📁 Updating moonshine service..."
# Remove existing moonshine folder and copy new one
if [ -d "$VENV_PATH/moonshine" ]; then
    echo "   Removing existing moonshine folder..."
    rm -rf "$VENV_PATH/moonshine"
fi

cp -r "$DESKTOP_SERVICES/moonshine" "$VENV_PATH/"
echo "✅ Moonshine service updated successfully"

echo "📁 Updating kokoro_onnx package..."
# Remove existing kokoro_onnx folder and copy new one
VENV_PACKAGES="$PROJECT_PATH/venv/lib/python3.12/site-packages"
if [ -d "$VENV_PACKAGES/kokoro_onnx" ]; then
    echo "   Removing existing kokoro_onnx folder..."
    rm -rf "$VENV_PACKAGES/kokoro_onnx"
fi

cp -r "$DESKTOP_PACKAGES/kokoro_onnx" "$VENV_PACKAGES/"
echo "✅ kokoro_onnx package updated successfully"

echo "📁 Updating moonshine_onnx package..."
# Remove existing moonshine_onnx folder and copy new one
if [ -d "$VENV_PACKAGES/moonshine_onnx" ]; then
    echo "   Removing existing moonshine_onnx folder..."
    rm -rf "$VENV_PACKAGES/moonshine_onnx"
fi

cp -r "$DESKTOP_PACKAGES/moonshine_onnx" "$VENV_PACKAGES/"
echo "✅ moonshine_onnx package updated successfully"

echo "📁 Updating kokoro_onnx dist-info..."
# Remove existing kokoro_onnx dist-info and copy new one
if [ -d "$VENV_PACKAGES/kokoro_onnx-0.4.9.dist-info" ]; then
    echo "   Removing existing kokoro_onnx-0.4.9.dist-info folder..."
    rm -rf "$VENV_PACKAGES/kokoro_onnx-0.4.9.dist-info"
fi

cp -r "$DESKTOP_PACKAGES/kokoro_onnx-0.4.9.dist-info" "$VENV_PACKAGES/"
echo "✅ kokoro_onnx dist-info updated successfully"

echo ""
echo "🔍 Verifying copied files..."

# Check assets
if [ -f "$PROJECT_PATH/assets/kokoro-v1.0.onnx" ]; then
    echo "✅ Kokoro model: $(du -h $PROJECT_PATH/assets/kokoro-v1.0.onnx | cut -f1)"
else
    echo "⚠️  Kokoro model not found in assets"
fi

if [ -f "$PROJECT_PATH/assets/voices-v1.0.bin" ]; then
    echo "✅ Voices file: $(du -h $PROJECT_PATH/assets/voices-v1.0.bin | cut -f1)"
else
    echo "⚠️  Voices file not found in assets"
fi

# Check services
if [ -f "$VENV_PATH/kokoro/tts.py" ]; then
    echo "✅ Kokoro TTS service installed"
else
    echo "❌ Kokoro TTS service not found"
fi

if [ -f "$VENV_PATH/moonshine/stt.py" ]; then
    echo "✅ Moonshine STT service installed"
else
    echo "❌ Moonshine STT service not found"
fi

# Check packages
if [ -d "$VENV_PACKAGES/kokoro_onnx" ]; then
    echo "✅ kokoro_onnx package installed"
else
    echo "❌ kokoro_onnx package not found"
fi

if [ -d "$VENV_PACKAGES/moonshine_onnx" ]; then
    echo "✅ moonshine_onnx package installed"
else
    echo "❌ moonshine_onnx package not found"
fi

if [ -d "$VENV_PACKAGES/kokoro_onnx-0.4.9.dist-info" ]; then
    echo "✅ kokoro_onnx dist-info installed"
else
    echo "❌ kokoro_onnx dist-info not found"
fi

echo ""
echo "🎉 All files copied successfully!"
echo ""
echo "📋 Summary:"
echo "   📁 Assets: $PROJECT_PATH/assets/"
echo "   🔊 Kokoro: $VENV_PATH/kokoro/"
echo "   🎤 Moonshine: $VENV_PATH/moonshine/"
echo ""
echo "🚀 Ready to test: python bot.py --transport webrtc"
echo ""
echo "💡 If you had issues, check the backup at: $PROJECT_PATH/assets.backup"