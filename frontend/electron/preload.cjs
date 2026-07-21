const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getAppInfo: () => ipcRenderer.invoke('app:get-info'),
  checkBackendHealth: () => ipcRenderer.invoke('backend:check-health'),
  isDesktop: true
});
