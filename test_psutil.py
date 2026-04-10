import psutil

prohibited = {
    'discord': 'Discord',
    'slack': 'Slack',
    'skype': 'Skype',
    'zoom': 'Zoom',
    'zoom.us': 'Zoom',
    'teams': 'Microsoft Teams',
    'whatsapp': 'WhatsApp',
    'telegram': 'Telegram',
    'anydesk': 'AnyDesk',
    'teamviewer': 'TeamViewer',
    'obs': 'OBS Studio',
    'quicktime player': 'QuickTime Player',
    'camtasia': 'Camtasia'
}

browsers = {
    'chrome': 'Google Chrome',
    'firefox': 'Firefox',
    'safari': 'Safari',
    'brave': 'Brave Browser',
    'edge': 'Microsoft Edge',
    'msedge': 'Microsoft Edge',
    'opera': 'Opera'
}

detected_apps = set()
for proc in psutil.process_iter(['name']):
    try:
        name = proc.info['name']
        if not name:
            continue
        name_lower = name.lower().replace('.exe', '').replace('.app', '')
        
        # Check prohibited apps
        for key, display_name in prohibited.items():
            if key == name_lower or key in name_lower:
                detected_apps.add(display_name)
                
        # Check browsers
        for key, display_name in browsers.items():
            if key == name_lower or key in name_lower:
                if key not in 'chrome':
                    detected_apps.add(display_name)
                    
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

print("Detected:", detected_apps)
