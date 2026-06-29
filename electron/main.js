const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

const PORT = process.env.DARKCLAW_PORT || 7430;
const URL  = `http://127.0.0.1:${PORT}`;

function createWindow() {
  const win = new BrowserWindow({
    width: 1440, height: 900,
    minWidth: 960, minHeight: 640,
    frame: false,
    backgroundColor: '#141109',
    icon: path.join(__dirname, '..', 'darkclaw.png'),
    title: 'DarkClaw',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  win.loadURL(URL);

  win.webContents.on('did-fail-load', (_e, code) => {
    if (code === -102 || code === -6)
      setTimeout(() => win.loadURL(URL), 400);
  });

  ipcMain.on('win:minimize', () => win.minimize());
  ipcMain.on('win:maximize', () => win.isMaximized() ? win.unmaximize() : win.maximize());
  ipcMain.on('win:close',    () => win.close());
  win.on('maximize',   () => win.webContents.send('win:maximized', true));
  win.on('unmaximize', () => win.webContents.send('win:maximized', false));
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
