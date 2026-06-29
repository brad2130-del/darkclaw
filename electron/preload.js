const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronShell', {
  minimize:    ()   => ipcRenderer.send('win:minimize'),
  maximize:    ()   => ipcRenderer.send('win:maximize'),
  close:       ()   => ipcRenderer.send('win:close'),
  onMaximized: (cb) => ipcRenderer.on('win:maximized', (_e, v) => cb(v)),
  isElectron:  true,
});
