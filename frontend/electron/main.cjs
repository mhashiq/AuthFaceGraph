const { app, BrowserWindow, systemPreferences, session, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

const fs = require('fs');

let mainWindow = null;
let pyBackendProcess = null;

const IS_DEV = process.env.NODE_ENV === 'development' || !app.isPackaged;
const PORT = 3000;
const BACKEND_PORT = 8000;
const BACKEND_HEALTH_URL = `http://127.0.0.1:${BACKEND_PORT}/api/v1/health`;

/**
 * Locate backend directory and python binary
 */
function getBackendPaths() {
  const possibleBackendDirs = [
    path.join(process.resourcesPath, 'backend'),
    path.resolve(__dirname, '../../backend'),
    '/Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend'
  ];

  let backendDir = possibleBackendDirs.find((d) => fs.existsSync(d)) || possibleBackendDirs[0];

  const possiblePythons = [
    path.join(backendDir, '.venv', 'bin', 'python'),
    path.join(backendDir, '.venv', 'bin', 'python3'),
    '/Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/.venv/bin/python',
    'python3',
    'python'
  ];

  let pythonExecutable = possiblePythons.find((p) => (p.startsWith('/') ? fs.existsSync(p) : true)) || 'python3';

  return { backendDir, pythonExecutable };
}

/**
 * Check if FastAPI backend is healthy and responding
 */
function checkBackendHealth() {
  return new Promise((resolve) => {
    http.get(BACKEND_HEALTH_URL, (res) => {
      resolve(res.statusCode === 200);
    }).on('error', () => {
      resolve(false);
    });
  });
}

/**
 * Poll backend until ready
 */
async function waitForBackend(maxAttempts = 30, interval = 1000) {
  for (let i = 0; i < maxAttempts; i++) {
    const ready = await checkBackendHealth();
    if (ready) return true;
    await new Promise((r) => setTimeout(r, interval));
  }
  return false;
}

/**
 * Launch FastAPI Backend Subprocess if not already running
 */
function startBackendProcess() {
  const { backendDir, pythonExecutable } = getBackendPaths();

  console.log(`[Electron] Starting FastAPI Backend using python: ${pythonExecutable}`);
  console.log(`[Electron] Backend Directory: ${backendDir}`);

  pyBackendProcess = spawn(
    pythonExecutable,
    ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)],
    {
      cwd: backendDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    }
  );

  pyBackendProcess.stdout.on('data', (data) => {
    console.log(`[FastAPI] ${data.toString().trim()}`);
  });

  pyBackendProcess.stderr.on('data', (data) => {
    console.error(`[FastAPI Error] ${data.toString().trim()}`);
  });

  pyBackendProcess.on('close', (code) => {
    console.log(`[FastAPI] Process exited with code ${code}`);
    pyBackendProcess = null;
  });
}

/**
 * Create main application window
 */
async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'AuthFaceGraph',
    titleBarStyle: 'hiddenInset',
    vibrancy: 'under-window',
    visualEffectState: 'active',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false // Allows accessing camera & local assets smoothly
    },
    show: false
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('[Electron] Failed to load:', errorCode, errorDescription);
  });

  // macOS Media Permission Handling
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media' || permission === 'camera' || permission === 'microphone') {
      callback(true);
    } else {
      callback(false);
    }
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  if (IS_DEV) {
    console.log('[Electron] Loading dev server:', `http://localhost:${PORT}`);
    await mainWindow.loadURL(`http://localhost:${PORT}`);
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    const indexPath = path.join(__dirname, '../dist/index.html');
    console.log('[Electron] Loading production build:', indexPath);
    await mainWindow.loadFile(indexPath);
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers
ipcMain.handle('app:get-info', () => ({
  name: 'AuthFaceGraph',
  platform: process.platform,
  arch: process.arch,
  electronVersion: process.versions.electron
}));

ipcMain.handle('backend:check-health', async () => {
  return await checkBackendHealth();
});

// App Lifecycle
app.whenReady().then(async () => {
  // Check if macOS system permissions for camera are granted
  if (process.platform === 'darwin') {
    try {
      const cameraStatus = await systemPreferences.askForMediaAccess('camera');
      console.log('[Electron] macOS Camera Permission Granted:', cameraStatus);
    } catch (e) {
      console.warn('[Electron] Camera permission error:', e);
    }
  }

  // Check if backend is already running (e.g. launched manually)
  const isBackendRunning = await checkBackendHealth();
  if (!isBackendRunning) {
    startBackendProcess();
    await waitForBackend();
  } else {
    console.log('[Electron] Backend already running on port', BACKEND_PORT);
  }

  await createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

function stopBackendProcess() {
  if (pyBackendProcess) {
    console.log('[Electron] Terminating FastAPI Backend Process...');
    pyBackendProcess.kill('SIGTERM');
    setTimeout(() => {
      if (pyBackendProcess) {
        pyBackendProcess.kill('SIGKILL');
      }
    }, 2000);
  }
}

app.on('window-all-closed', () => {
  stopBackendProcess();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendProcess();
});
